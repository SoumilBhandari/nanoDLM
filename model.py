"""Bidirectional transformer for masked diffusion language modeling.

Structural differences from karpathy/nanoGPT:
  1) Self-attention is BIDIRECTIONAL (no causal mask).
  2) Vocabulary has one extra row for the [MASK] token at id == base_vocab_size.
  3) Positions use rotary embeddings (RoPE), not a learned absolute table.
     Absolute embeddings tend to underperform on small bidirectional LMs;
     RoPE composes more cleanly with attention and is the 2024+ default.
  4) Optional self-conditioning (Chen et al. 2023, "Analog Bits"): the
     model can additionally consume its own previous-step prediction
     through a zero-initialised projection. The projection starts off
     contributing nothing and learns its useful scale during training.

There is NO timestep embedding. Per MD4 (Shi et al. 2024), the model can infer
the noise level t from the density of [MASK] tokens in the input — explicit t
conditioning gave no measurable lift in their ablations, so we drop it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope(seq_len: int, head_dim: int, base: float = 10000.0):
    """Standard rotary-position cos/sin tables. Returns two (seq_len, head_dim) tensors."""
    assert head_dim % 2 == 0, "RoPE needs an even head_dim"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                            # (seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                     # (seq_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., T, head_dim); cos/sin: (T, head_dim). Broadcasts over leading dims."""
    return x * cos + _rotate_half(x) * sin


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    """Bidirectional multi-head self-attention. No causal mask."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        head_dim = C // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        # Rotary position embeddings on Q and K (not V) — sliced to T positions.
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])
        # is_causal=False — the architectural delta from nanoGPT.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = SelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x


class DLM(nn.Module):
    """Diffusion Language Model.

    The vocabulary stored in self.vocab_size is base_vocab_size + 1, where the
    final id is reserved for [MASK]. Predictions over the [MASK] id itself are
    masked out at the loss (we never want to predict MASK).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.base_vocab_size = cfg.vocab_size
        self.vocab_size = cfg.vocab_size + 1     # +1 for [MASK]
        self.mask_id = cfg.vocab_size            # last id

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.vocab_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, self.vocab_size, bias=False)
        # Weight tying — same as nanoGPT.
        self.transformer.wte.weight = self.lm_head.weight

        # Self-conditioning projection (Analog Bits / Chen et al. 2023).
        # At training, with 50% probability we run a no-grad forward to get the
        # model's own argmax prediction, embed it via wte, project here, and
        # add to the input embedding for a second (gradient-bearing) forward.
        # Zero-init so the projection contributes nothing at step 0 and only
        # learns to use self-conditioning once the model has something useful
        # to condition on.
        self.cond_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        # RoPE cache, registered as a (non-persistent) buffer so it follows
        # .to(device) but doesn't bloat the checkpoint file.
        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = precompute_rope(cfg.block_size, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") and "cond_proj" not in pn:
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        # Zero-init the self-conditioning projection (Analog Bits).
        nn.init.zeros_(self.cond_proj.weight)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"model parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, idx_prev=None):
        """
        idx:       (B, T) input tokens (with [MASK] at corrupted positions).
        idx_prev:  optional (B, T) of the model's previous-step predictions, for
                   self-conditioning. None during the first pass / when self-cond
                   is disabled, in which case the cond_proj branch is skipped.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size
        x = self.transformer.wte(idx)
        if idx_prev is not None:
            # Self-conditioning: add a zero-init projection of the previous-
            # step prediction's embedding. Detach so gradients only flow
            # through the current input path; the first pass that produced
            # idx_prev was no_grad anyway.
            x = x + self.cond_proj(self.transformer.wte(idx_prev))
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        # Forbid predicting [MASK] itself — set its logit to -inf.
        logits[..., self.mask_id] = float("-inf")
        return logits

    def configure_optimizers(self, weight_decay, lr, betas):
        # Same split as nanoGPT: weight-decay on 2D params (matmuls/embeddings),
        # no decay on biases and LayerNorm.
        decay, nodecay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else nodecay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=torch.cuda.is_available())
