# nanoDLM

The simplest possible diffusion language model. Train a tiny LLaDA-style masked-diffusion LM on tiny Shakespeare and watch text crystallize from noise in your terminal.

If [nanoGPT](https://github.com/karpathy/nanoGPT) is the simplest *autoregressive* LLM, this is the simplest *diffusion* LLM. ~600 lines of pure PyTorch, no `diffusers`, no HuggingFace Trainer, no tokenizer. Char-level. One afternoon to read end-to-end.

## Why this exists

In 2026, diffusion language models stopped being a curiosity. Inception's Mercury 2 ships at 800–1000 tok/s. LLaDA-8B is competitive with autoregressive LMs at the same scale. But every educational repo on the internet still teaches autoregressive LMs. The "what's actually different?" answer is shorter than people think — it's about five lines of code on top of nanoGPT. This repo is those five lines, in context.

## Quickstart

```bash
pip install torch numpy
python prepare.py        # downloads ~1MB of Shakespeare, builds char vocab
python train.py          # ~30 min on a 4090, longer on CPU/MPS
python sample.py --verbose --steps 64
```

Watch the terminal: every step replaces some `_` placeholders with characters until you have plausible Shakespeare.

## The five deltas from nanoGPT

That's the whole pedagogical payload. Everything else is copy-paste.

| # | Change | File | Lines |
|---|---|---|---|
| 1 | Drop the causal mask in self-attention | `model.py` | 1 |
| 2 | Vocab += 1 for a `[MASK]` token | `model.py` | 3 |
| 3 | Forward signature unchanged — no timestep input | `model.py` | 0 |
| 4 | Loss = 1/t-weighted CE on masked positions | `train.py` | 8 |
| 5 | Sampler = iterative low-confidence remasking | `sample.py` | 15 |

### The training step, in full

```python
t = torch.rand(B, 1, device=x.device).clamp(min=1e-3)   # per-sample noise level
mask = torch.rand(B, T, device=x.device) < t            # Bernoulli(t) per token
x_t = torch.where(mask, MASK_ID, x)                     # corrupt

logits = model(x_t)                                     # (B, T, V)
loss_tok = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1),
                           reduction="none").view(B, T)
loss = (loss_tok * mask / t).sum() / mask.sum().clamp(min=1)
```

That is the entire diffusion-LM training objective. No score networks, no Gaussian noise, no embedding-space tricks. Just BERT with a random mask ratio and a 1/t weight. The 1/t weight is the absorbing-state ELBO — see Sahoo et al. 2024 §3.

### The sampler, in full

```python
x = torch.full((1, length), MASK_ID, device=device)
for i in range(steps):
    logits = model(x); probs = logits.softmax(-1)
    conf, pred = probs.max(-1)
    n_keep_masked = int((1 - (i+1)/steps) * length)
    is_masked = (x == MASK_ID)
    conf_masked = conf.masked_fill(~is_masked, -float("inf"))
    kth = conf_masked.kthvalue(n_keep_masked + 1, dim=-1).values
    x = torch.where(conf_masked >= kth.unsqueeze(-1), pred, x)
```

At each step we predict every position but only commit the highest-confidence ones. The rest stay masked and get revisited next step.

## The lesson: more steps = better samples

```bash
python sample.py --ablate
```

Sweeps `steps ∈ {1, 4, 16, 64, 256}` on a fixed prompt. With `steps=1` you get BERT-style independent samples — locally plausible characters that don't form words. With `steps=64` you get coherent Shakespearean clauses. The quality-vs-steps curve *is* the lesson — iterative refinement is doing real work.

## Expected numbers

On a single 4090, default config (~10M params, 6 layers, 384 dim, block 256):

| Metric | Value |
|---|---|
| Wallclock | ~30 min |
| Final val loss | ~1.4–1.5 nats/char |
| nanoGPT char baseline (same wallclock) | ~1.3 nats/char |
| Loss tax for going non-AR | ~10–15% |

Diffusion LMs pay a small NLL tax in exchange for parallel decoding and a richer sampling story. This is expected and is the trade-off Mercury and LLaDA accept at scale.

## Explicit non-goals

This repo deliberately **does not** include:

- Classifier-free guidance (CFG is for instruct models)
- Timestep embedding (per [MD4](https://arxiv.org/abs/2406.04329) ablations, not needed — the model infers t from mask density)
- KV cache / block-wise semi-autoregressive decoding (Mercury's speed trick)
- SEDD score-entropy alternative (more general but obscures the core idea)
- Continuous-time ELBO derivation in code (read [Sahoo et al.](https://arxiv.org/abs/2406.07524) §3)
- SFT, instruction tuning, RLHF
- DDP / FSDP / multi-GPU
- BPE or any tokenizer (char-level only)
- Fancy noise schedules — linear `t ~ U(0,1)` is provably optimal for the absorbing case

If you want any of these, fork it. The point of this repo is the five deltas, not the full feature surface.

## File map

```
nanoDLM/
├── config.py    single dataclass, ~30 lines
├── prepare.py   tiny shakespeare → bin, ~40 lines
├── model.py     bidirectional transformer, ~140 lines
├── train.py     masked-diffusion training loop, ~150 lines
├── sample.py    iterative remasking sampler, ~80 lines
└── README.md    you are here
```

## References

- **LLaDA** — Nie et al. 2025, *Large Language Diffusion Models* — primary recipe reference, low-confidence remasking sampler.
- **MD4** — Shi et al. 2024, *Simplified and Generalized Masked Diffusion for Discrete Data* — shows timestep embedding is unnecessary.
- **Sahoo et al.** — 2024, *Simple and Effective Masked Diffusion Language Models* — clean ELBO derivation, the 1/t weight.
- **SEDD** — Lou et al. 2024, *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution* — the alternative we didn't pick.
- **Mercury** — Inception Labs 2025–2026 — production deployment.

## Credit

The transformer code, init scheme, and training scaffolding are lifted nearly verbatim from [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). The diffusion bits are the contribution.
