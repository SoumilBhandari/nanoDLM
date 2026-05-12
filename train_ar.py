"""Train the autoregressive baseline.

Same hyperparameters as train.py (MDM), so the comparison is apples-to-apples
at matched architecture and matched token-throughput. The only structural
differences:

  - No diffusion noise/mask: each batch is just (x, y) with y = x shifted by 1.
  - Loss is standard next-token cross-entropy.
  - No self-conditioning, no two-pass step (so each batch is ~1.5x faster
    than the MDM's mixed 1-pass / 2-pass batches).

Output: out/ckpt_ar.pt — picked up by eval.py for the head-to-head table.
"""
import math
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch

from config import Config
from model_ar import ARLM

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
print(f"vocab size: {cfg.vocab_size}")

train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    """Returns (x, y) with y the next-token target of x (shifted by 1)."""
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# model ---------------------------------------------------------------------
model = ARLM(cfg).to(device)
optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2))
if cfg.compile and device == "cuda":
    try:
        import triton  # noqa: F401
        print("compiling model...")
        model = torch.compile(model)
    except ImportError:
        print("triton not installed; skipping torch.compile (training will run, just slower)")


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(split)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def sample_ar(prompt_ids, length=128, temperature=1.0, top_p=0.9):
    """Greedy-like autoregressive sample for the in-training preview."""
    model.eval()
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None]
    for _ in range(length):
        idx_cond = x[:, -cfg.block_size:]
        logits = m(idx_cond)[:, -1, :] / temperature
        probs = logits.softmax(-1)
        # Top-p truncation (same trick as sample.py)
        if top_p < 1.0:
            sp, si = probs.sort(dim=-1, descending=True)
            cum = sp.cumsum(dim=-1)
            drop = (cum - sp) > top_p
            sp = sp.masked_fill(drop, 0.0)
            probs = torch.zeros_like(probs).scatter_(-1, si, sp)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        nxt = torch.multinomial(probs, 1)
        x = torch.cat([x, nxt], dim=1)
    model.train()
    return x


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
        out = sample_ar([0], length=128)        # prompt = newline
        print("--- AR sample ---")
        print("".join(itos[int(i)] for i in out[0].tolist()))
        print("-----------------")

    if step == cfg.max_steps:
        break

    x, y = get_batch("train")
    with ctx:
        _, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

# save ----------------------------------------------------------------------
m = model._orig_mod if hasattr(model, "_orig_mod") else model
ckpt_path = os.path.join(cfg.out_dir, "ckpt_ar.pt")
torch.save({"model": m.state_dict(), "config": cfg.__dict__}, ckpt_path)
print(f"saved {ckpt_path}")
