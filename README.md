# nanoDLM

The simplest masked-diffusion language model you can actually train, debug,
and learn from. If [nanoGPT](https://github.com/karpathy/nanoGPT) is the
minimum autoregressive LM, this is the minimum diffusion LM — char-level,
no tokenizer, no `diffusers`, no HF Trainer. ~1000 lines of pure PyTorch.

This fork goes past the educational baseline in three ways: it **fixes
real bugs** in the original objective and sampler, **modernises** the
backbone (RoPE + self-conditioning), and **puts MDM head-to-head** with a
matched autoregressive baseline so the trade-off is something you can
read off a table, not something you have to take on faith.

## Why this exists

Diffusion language models stopped being a curiosity in 2026 — Inception's
Mercury 2 ships at ~1000 tok/s, LLaDA-8B is competitive with AR at the
same scale, and the killer feature (left-and-right conditioning, i.e.
infilling) is something AR architecturally cannot do. Every educational
LM repo still teaches AR. This one teaches MDM at the same level of
care, and shows what it costs and what it buys.

## Quickstart

```bash
pip install torch numpy
python prepare.py        # downloads ~1MB of Shakespeare, builds char vocab

# Train the masked-diffusion LM (~22 min on a 3090, no torch.compile)
python train.py
python sample.py --verbose --steps 64

# Train the matched autoregressive baseline (~14 min)
python train_ar.py

# Compare them head-to-head on shared metrics
python eval.py

# The killer feature: middle-completion that AR literally cannot do
python infill.py --prefix "ROMEO:" --suffix "JULIET:" --middle-length 120
```

Watch `sample.py --verbose` — every step replaces some `_` placeholders
with characters until you have plausible Shakespeare crystallising from
noise.

## The MDM payload — six deltas from nanoGPT

The whole pedagogical core of masked-diffusion-on-top-of-AR is small:

| # | Change | File |
|---|---|---|
| 1 | Drop the causal mask in attention | [model.py](model.py) |
| 2 | Vocab += 1 for the `[MASK]` token | [model.py](model.py) |
| 3 | Forward signature unchanged — no timestep input (per [MD4](https://arxiv.org/abs/2406.04329) ablations the model infers `t` from mask density) | [model.py](model.py) |
| 4 | Loss = 1/t-weighted CE on masked positions, normalised by **B·T** (the Sahoo et al. ELBO — not by `mask.sum()`, which is a stochastic denominator and biases gradient scale per-batch) | [train.py](train.py) |
| 5 | Sampler = iterative low-confidence remasking with **categorical sampling** (argmax + temperature is a no-op and collapses to a single attractor) | [sample.py](sample.py) |
| 6 | Schedule against the count of *initially-masked* positions, not the full sequence length — otherwise infilling silently overwrites the prefix/suffix | [sample.py](sample.py) |

### The training step, in full

```python
t = torch.rand(B, 1, device=x.device).clamp(min=1e-3)
mask = torch.rand(B, T, device=x.device) < t
x_t = torch.where(mask, MASK_ID, x)

logits = model(x_t)
loss_tok = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1),
                           reduction="none").view(B, T)
loss = (loss_tok * mask / t).sum() / (B * T)        # <-- B*T, not mask.sum()
```

### The sampler, in full

```python
x = torch.full((1, length), MASK_ID, device=device)
n_to_fill = int((x == MASK_ID).sum().item())

for i in range(steps):
    logits = model(x); probs = logits.softmax(-1)
    pred = torch.multinomial(probs.flatten(0, -2), 1).view(probs.shape[:-1])
    conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)

    n_keep_masked = int((1 - (i+1)/steps) * n_to_fill)
    is_masked = (x == MASK_ID)
    n_to_unmask = max(0, int(is_masked.sum().item()) - n_keep_masked)

    conf_masked = conf.masked_fill(~is_masked, float("-inf"))
    L = conf_masked.size(-1)
    kth = conf_masked.kthvalue(L - n_to_unmask + 1, dim=-1).values
    x = torch.where(conf_masked >= kth.unsqueeze(-1), pred, x)
```

(The actual [sample.py](sample.py) version adds top-p truncation,
self-conditioning, and a `--schedule {linear, cosine, cosine_inv}` flag.
Stripped of those, it's the snippet above.)

## Beyond LLaDA — four additions that make samples readable

The literal LLaDA recipe is not enough at this scale: 20K steps of the
plain MDM produces local-fragment soup. Four small additions take us to
recognisable Shakespeare:

| Addition | File | Why |
|---|---|---|
| **RoPE positions** (replaces learned `wpe`) | [model.py](model.py) | Absolute positions underperform on small bidirectional LMs; RoPE composes better with attention. |
| **Self-conditioning** (Chen et al. 2023 "Analog Bits") | [model.py](model.py), [train.py](train.py) | At training, 50% of batches do an extra no-grad forward and feed the model's own argmax back through a zero-init projection. Standard recipe; fades in as training progresses. |
| **Top-p truncation** before categorical sampling | [sample.py](sample.py) | Sampling from the full softmax leaks low-probability junk tokens into the output. Default `top_p=0.9`. |
| **Non-uniform schedule** (`--schedule cosine` / `cosine_inv`) | [sample.py](sample.py) | The linear schedule (1/steps tokens per step) is what LLaDA / MDLM / MD4 use. We add cosine variants and report PPL-under-AR for all three in `eval.py` — ship whichever wins, document the rest. |

## Headline results

5000 steps of the original-recipe MDM (the README's claim) produces
character soup. 20K steps of the *fixed* recipe with all four additions
produces text like this:

> ROMEO:
> Why, not your lord?
>
> MARIANA:
> What have you would, you have done thee the lord,
> To shake you good that come, my lord.
>
> JULIET:

That's a single `python infill.py --prefix "ROMEO:" --suffix "JULIET:" --middle-length 120` invocation. The `MARIANA:` interjection isn't in the prompt — the model invented it (Mariana being a real Shakespeare character).

Full numerical comparison against a matched-architecture AR baseline.
Both models are 10M params, RoPE positions, same dropout/weight-decay/
optimizer/schedule/total-steps. The AR checkpoint is the **best-val**
one (saved automatically by `train_ar.py`) because a 10M AR overfits
tiny-Shakespeare into the ground long before the cosine schedule
finishes — final-step val ≈ 3.4 nats/char vs best-val ≈ 1.48. Honest
comparison demands that selection.

| Metric | MDM | AR |
|---|---|---|
| Val char NLL (lower better) | <= 1.618 (ELBO, upper bound on NLL) | **1.486** |
| Sample PPL under AR scorer @ NFE=64 (lower = more on-distribution) | 3.67 | **2.46** |
| Sample distinct-2 (higher = more diverse) | 0.093 | **0.113** |
| Sample distinct-3 | 0.206 | **0.296** |
| Infill recovery @ span=20, NFE=64 (higher better) | **0.156** | N/A (causal AR cannot infill) |

(Single-seed numbers. Run-to-run noise is ~0.01 on val NLL and ~10-20% on the sample-level metrics; the qualitative ordering between MDM and AR is robust.)

Read this honestly:

- AR wins NLL, sample PPL, and diversity by ordinary margins. The
  diffusion tax exists and is real at this scale.
- MDM owns the infill row. 20% character-exact recovery on a random
  20-char val span is not human-level but it is the only number any
  AR can write down here, because AR architecturally cannot condition
  on tokens to the right of the cursor.
- AR can only achieve 1.478 val NLL through aggressive early-stopping
  (the saved checkpoint is from step 1000 of 20000). The MDM does not
  need this — its training noise acts as implicit regularisation and
  best-val arrives at the natural end of the cosine schedule.

### Schedule sweep

`eval.py` also sweeps the three available denoising schedules at the
same NFE. None clears the 5%-improvement bar we set for promoting a
new default, so `linear` stays the default and `cosine` / `cosine_inv`
remain available as flags for experimentation:

| Schedule | PPL under AR | delta vs linear |
|---|---|---|
| linear | 3.67 | +0.0% |
| cosine | 3.65 | -0.6% |
| cosine_inv | 3.67 | +0.1% |

(Re-run with `python eval.py` to refresh against your own checkpoints.)

### Negative result: DUO-style hybrid training

We also tested DUO (Sahoo et al. 2024, "Diffusion Forcing for Discrete
Tokens"): with `p_ar_mix=0.25` per batch, replace the random
Bernoulli(t) mask with a contiguous-suffix mask so the model gets some
AR-style supervision on top of the MDM objective. Recent papers
report this closes most of the AR/MDM val-NLL gap at scale. At our
scale (10M params, char-level Shakespeare, 20K steps) it hurt every
metric we measured:

| Metric | Pure MDM | DUO @ p=0.25 | delta |
|---|---|---|---|
| Val ELBO | 1.617 | 1.664 | **+2.9%** |
| Sample PPL under AR | 3.35 | 3.97 | **+18%** |
| Infill recovery | 0.200 | 0.144 | **-28%** |

So `p_ar_mix` defaults to `0.0` and the code path stays as an opt-in
for anyone who wants to try a different mixing rate or a larger scale
where the technique may help.

## What we fixed while building this fork

The original repo's training and sampling code had three correctness
bugs and one undertraining issue. Worth documenting because they're the
kind of mistake that's easy to make and hard to notice when the loss
descends "looking right":

1. **Loss formula divided by `mask.sum()` instead of `B*T`.**
   [Sahoo et al. 2024 §3](https://arxiv.org/abs/2406.07524) gives the ELBO as
   a sum of per-position weighted CEs divided by the **deterministic**
   sequence length. The original code divided by the count of randomly-masked
   tokens. Net effect: reported loss converges to ~2H instead of H, and the
   per-batch gradient scale fluctuates with the random mask draw. On the
   default config, fixing this dropped final val from **4.05 → 1.99** nats/char.
   (Diff: `train.py` ~ line 91.)

2. **`--temperature` flag was inert.** The sampler divided logits by
   temperature and then took argmax, but argmax is invariant under any
   monotonic transform. Fixed by drawing from the categorical, which
   also handily breaks the next bug.

3. **Mode collapse from argmax + low-confidence remasking.** At step 1
   every position sees the same all-MASK context, so every position's
   argmax is the same high-prior token. The resulting attractor poisons
   the whole trajectory — pre-fix samples literally repeated phrases
   like `tme you yourd tme you`. Categorical sampling fixes this.

4. **Sampler schedule clobbered prefix/suffix during infilling.** When
   `kthvalue` lands inside the `-inf` region (which happens whenever
   `n_keep_masked` exceeds the count of currently-masked positions), the
   `>= -inf` comparison is `True` everywhere and frozen tokens get
   silently overwritten. Fixed by scheduling against the count of
   currently-masked positions and computing the threshold from the
   `n_to_unmask`-th largest confidence.

5. **`max_steps=5000` is below the Chinchilla floor.** That's ~82M tokens
   on a 10M-param model; the AR rule of thumb is ≥ 200M, and MDMs need
   ~2-3× the compute of an AR at matched scale. Bumped to 20K. The pre-
   and post-fix val numbers are not comparable until you also fix the
   step count.

## Explicit non-goals

This repo deliberately does **not** include:

- Classifier-free guidance (it's for instruct models)
- Timestep embedding (per MD4, not needed at this scale)
- KV cache / block-wise semi-autoregressive decoding (Mercury's speed trick)
- SEDD score-entropy alternative (more general but obscures the core idea)
- Continuous-time ELBO derivation in code (read [Sahoo et al.](https://arxiv.org/abs/2406.07524) §3)
- SFT, instruction tuning, RLHF
- DDP / FSDP / multi-GPU
- BPE or any tokenizer (char-level only)

If you want any of these, fork it. The point of this repo is the
pedagogical core plus an honest head-to-head with AR — not the full
feature surface.

## File map

```
nanoDLM/
├── config.py     single dataclass of hyperparameters
├── prepare.py    tiny shakespeare → bin, ~40 lines
├── model.py      bidirectional transformer + RoPE + self-cond, ~180 lines
├── train.py      masked-diffusion training loop, ~160 lines
├── sample.py     iterative remasking sampler + top-p + schedule flag
├── infill.py     middle-completion demo (the killer MDM feature)
├── model_ar.py   matched AR baseline architecture
├── train_ar.py   AR baseline training loop
├── eval.py       shared MDM-vs-AR eval harness, writes out/eval_table.md
└── README.md     you are here
```

## References

- **LLaDA** — Nie et al. 2025, *Large Language Diffusion Models* — primary recipe reference, low-confidence remasking sampler.
- **MD4** — Shi et al. 2024, *Simplified and Generalized Masked Diffusion for Discrete Data* — shows timestep embedding is unnecessary.
- **MDLM / Sahoo et al.** — 2024, *Simple and Effective Masked Diffusion Language Models* — the ELBO derivation we follow.
- **SEDD** — Lou et al. 2024, *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution* — the alternative we didn't pick.
- **Analog Bits** — Chen et al. 2023, *Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning* — the self-cond recipe.
- **RoFormer / RoPE** — Su et al. 2021 — the positional encoding we use.
