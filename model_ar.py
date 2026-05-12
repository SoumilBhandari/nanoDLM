"""Autoregressive transformer baseline — identical architecture to model.DLM
except attention is causal and the prediction target is shift-by-one.

This file exists so we can do a real head-to-head between an MDM and an AR
LM at matched architecture, parameter count, and total training compute.
That comparison is the actual lesson of this repo: not "here is how MDM
works in isolation," but "here is what diffusion buys you, and what it
costs, relative to the autoregressive default."

Everything reusable (LayerNorm, MLP, RoPE helpers) is imported from
model.py. Only the parts that genuinely differ — causal attention and the
top-level wrapper — live here.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LayerNorm, MLP, precompute_rope, apply_rope


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with RoPE. The only delta from
    model.SelfAttention is is_causal=True."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
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
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class CausalBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x


class ARLM(nn.Module):
    """Autoregressive Language Model.

    No [MASK] token (vocab_size matches the data's char vocab exactly), no
    self-conditioning, no diffusion. forward(idx, targets) returns either
    just logits (targets=None) or (logits, loss) where loss is standard
    next-token cross-entropy — same convention as nanoGPT.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size  # no +1: nothing to mask

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.vocab_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([CausalBlock(cfg) for _ in range(cfg.n_layer)]),
            ln_f=LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, self.vocab_size, bias=False)
        # Weight tying.
        self.transformer.wte.weight = self.lm_head.weight

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = precompute_rope(cfg.block_size, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"AR model parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        x = self.transformer.drop(self.transformer.wte(idx))
        for block in self.transformer.h:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
        )
        return logits, loss

    def configure_optimizers(self, weight_decay, lr, betas):
        decay, nodecay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else nodecay).append(p)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": nodecay, "weight_decay": 0.0}],
            lr=lr, betas=betas, fused=torch.cuda.is_available(),
        )
