"""Train a masked-diffusion language model on tiny Shakespeare.

The pedagogical payload is the eight-line training step in `loss_fn` below:
sample a per-example noise level t, mask each token independently with prob t,
predict the masked tokens, and weight the cross-entropy by 1/t.

Everything else is standard nanoGPT-style scaffolding.
"""
import math
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch

from config import Config
from model import DLM
from sample import generate

cfg = Config()
torch.manual_seed(cfg.seed)
os.makedirs(cfg.out_dir, exist_ok=True)

# device / dtype ------------------------------------------------------------
if cfg.device == "cuda" and not torch.cuda.is_available():
    cfg.device = "mps" if torch.backends.mps.is_available() else "cpu"
device = cfg.device
dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
ptdtype = dtype_map[cfg.dtype if device == "cuda" else "float32"]
ctx = (torch.amp.autocast(device_type="cuda", dtype=ptdtype)
       if device == "cuda" else nullcontext())
print(f"device: {device}   dtype: {ptdtype}")

# data ----------------------------------------------------------------------
data_dir = os.path.join(os.path.dirname(__file__), cfg.data_dir)
with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
cfg.vocab_size = meta["vocab_size"]
itos = meta["itos"]
print(f"vocab size: {cfg.vocab_size} (+1 [MASK])")

train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + cfg.block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    return x


# model ---------------------------------------------------------------------
model = DLM(cfg).to(device)
MASK_ID = model.mask_id
optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2))
if cfg.compile and device == "cuda":
    try:
        import triton  # noqa: F401
        print("compiling model...")
        model = torch.compile(model)
    except ImportError:
        print("triton not installed; skipping torch.compile (training will run, just slower)")


def loss_fn(model, x, self_cond_prob: float = 0.5):
    """The whole pedagogical payload of nanoDLM.

    Self-conditioning (Chen et al. 2023): with probability self_cond_prob,
    do an extra no-grad forward to get the model's own argmax prediction
    and feed it back as idx_prev on the gradient-bearing forward. Otherwise
    idx_prev is None and the cond_proj path is a no-op (zero-init). At
    matched param count this is documented to lift MDM val by ~0.05-0.15
    nats. Cost: ~50% wallclock since half the batches do 2 forwards.
    """
    B, T = x.shape
    t = torch.rand(B, 1, device=x.device).clamp(min=cfg.eps)   # per-sample noise
    mask = torch.rand(B, T, device=x.device) < t               # Bernoulli(t)
    x_t = torch.where(mask, MASK_ID, x)                        # corrupt

    idx_prev = None
    if torch.rand(()).item() < self_cond_prob:
        # No-grad first pass produces the self-conditioning input. argmax is
        # safe because logits[..., MASK_ID] is -inf so MASK is never picked.
        with torch.no_grad():
            idx_prev = model(x_t).argmax(dim=-1)

    logits = model(x_t, idx_prev)                              # (B, T, V_with_mask)
    loss_tok = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        x.reshape(-1),
        reduction="none",
    ).view(B, T)
    # Sahoo et al. 2024 §3 ELBO: divide by the deterministic constant B*T,
    # not by the stochastic mask.sum(). With the stochastic denominator the
    # reported loss is ~2H instead of H, the small-t signal gets diluted by
    # heavily-masked siblings in the same batch, and per-batch gradient scale
    # depends on the random mask draw. The proper estimator below is the
    # standard MDM-NELBO and converges to nats/char.
    return (loss_tok * mask / t).sum() / (B * T)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            with ctx:
                # Disable self-conditioning at eval so val loss is directly
                # comparable to runs without self-cond. The deployment-time
                # behaviour (self-cond always on) is what `sample.py` exercises.
                losses[k] = loss_fn(model, get_batch(split), self_cond_prob=0.0)
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(step):
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.1 + 0.5 * (cfg.lr - cfg.lr * 0.1) * (1 + math.cos(math.pi * progress))


# training loop -------------------------------------------------------------
print(f"training for {cfg.max_steps} steps")
t0 = time.time()
for step in range(cfg.max_steps + 1):
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    if step % cfg.eval_interval == 0:
        losses = estimate_loss()
        dt = time.time() - t0
        print(f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} | "
              f"lr {lr:.2e} | {dt:6.1f}s")

    if step > 0 and step % cfg.sample_interval == 0:
        print("--- sample (steps=64) ---")
        m = model._orig_mod if hasattr(model, "_orig_mod") else model
        out = generate(m, length=128, steps=64, device=device, verbose=False)
        print("".join(itos[int(i)] for i in out[0].tolist()))
        print("-------------------------")

    if step == cfg.max_steps:
        break

    x = get_batch("train")
    with ctx:
        loss = loss_fn(model, x)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

# save ----------------------------------------------------------------------
m = model._orig_mod if hasattr(model, "_orig_mod") else model
ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
torch.save({"model": m.state_dict(), "config": cfg.__dict__}, ckpt_path)
print(f"saved {ckpt_path}")
