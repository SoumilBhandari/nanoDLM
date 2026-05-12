"""Iterative low-confidence remasking sampler (LLaDA-style).

Start from a sequence of all-[MASK] tokens. At each step, predict every
position, but only commit the highest-confidence predictions; the rest stay
masked and are revisited next step. This is the magic moment of nanoDLM —
print the partially-denoised string each step and watch text crystallize.
"""
import argparse
import os
import pickle

import torch

from config import Config
from model import DLM


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out probs outside the top-p (nucleus) set, renormalise.

    Implementation note: the standard "shift-by-one" trick. We drop a token if
    the cumulative-probability *not including it* already exceeds top_p; this
    guarantees the top-1 is always kept (its prefix-cum is 0) and a token whose
    inclusion pushes us past the threshold for the first time is still kept.
    """
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cum = sorted_probs.cumsum(dim=-1)
    drop_sorted = (cum - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(drop_sorted, 0.0)
    out = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    return out / out.sum(dim=-1, keepdim=True).clamp(min=1e-12)


@torch.no_grad()
def generate(model, length=256, steps=64, temperature=1.0, top_p=0.9, device="cpu",
             verbose=False, itos=None, x_init=None, self_cond=True):
    """Generate via `steps` rounds of low-confidence remasking.

    Two modes:
      - unconditional: start from all-MASK of `length` tokens (the default).
      - conditional (infilling): pass `x_init` of shape (1, L) with some
        positions already filled in (non-MASK). Those positions are frozen
        for the whole trajectory; only the [MASK]-id positions get denoised.
        This is what `infill.py` uses to demonstrate the bidirectional
        prefix+suffix → middle completion that AR models cannot do.

    `top_p` truncates the categorical to the nucleus before sampling; without
    this, low-probability junk tokens leak in and shred local coherence. 0.9
    is the standard default; 1.0 disables.

    `self_cond` toggles inference-time self-conditioning (Chen et al. 2023).
    With it, the previous step's argmax prediction is fed back through the
    model's `cond_proj` on every step after the first. Trained checkpoints
    that used self-cond at training benefit from having it on at inference.
    For checkpoints from a no-self-cond training, this is a near no-op
    because cond_proj remains near zero.
    """
    model.eval()
    if x_init is not None:
        x = x_init.to(device).clone()
    else:
        x = torch.full((1, length), model.mask_id, dtype=torch.long, device=device)

    # Schedule against the count of *initially*-masked positions, not the
    # full sequence length. Otherwise the first ~(1 - n_masked/L) fraction
    # of steps are no-ops when most of the sequence is pre-filled.
    n_to_fill = int((x == model.mask_id).sum().item())
    idx_prev = None  # self-conditioning input, populated after the first step

    for i in range(steps):
        logits = model(x, idx_prev if self_cond else None) / temperature
        probs = _top_p_filter(logits.softmax(-1), top_p)
        # Categorical sampling at every position, not argmax. Two reasons:
        # (1) argmax is invariant to monotonic transforms of the logits, so
        #     `temperature` was previously a no-op — this makes it meaningful.
        # (2) argmax + low-confidence remasking is the textbook source of
        #     mode collapse in MDM sampling: identical all-MASK context at
        #     step 1 makes every position vote for the same high-prob token,
        #     and the resulting attractor poisons the whole trajectory. The
        #     MDLM / MD4 samplers all draw from the categorical.
        # Confidence is then the probability *of the sampled token*, which
        # is the principled signal for low-confidence remasking.
        B, T, V = probs.shape
        pred = torch.multinomial(probs.reshape(B * T, V), 1).view(B, T)
        conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        # Self-conditioning carries `pred` forward to the next step.
        idx_prev = pred

        # Schedule: how many tokens should still be MASKed after this step.
        frac_remaining = 1.0 - (i + 1) / steps
        n_keep_masked = int(frac_remaining * n_to_fill)

        is_masked = (x == model.mask_id)
        n_masked_now = int(is_masked.sum().item())
        n_to_unmask = max(0, n_masked_now - n_keep_masked)

        # -inf at non-mask positions so they can never be chosen for unmasking.
        conf_masked = conf.masked_fill(~is_masked, float("-inf"))

        if n_to_unmask == 0:
            unmask = torch.zeros_like(is_masked)
        elif n_to_unmask >= n_masked_now:
            unmask = is_masked
        else:
            # Pick the n_to_unmask highest-confidence currently-masked positions.
            # kthvalue(k) returns the k-th SMALLEST; to get the n_to_unmask-th
            # largest we ask for k = L - n_to_unmask + 1. The threshold is
            # guaranteed finite (n_to_unmask <= n_masked_now ⇒ enough finite
            # entries above the -inf floor at non-mask positions).
            L = conf_masked.size(-1)
            kth = conf_masked.kthvalue(L - n_to_unmask + 1, dim=-1).values
            unmask = conf_masked >= kth.unsqueeze(-1)

        x = torch.where(unmask, pred, x)

        if verbose and itos is not None:
            display = "".join(
                itos[int(t)] if t != model.mask_id else "_" for t in x[0].tolist()
            )
            print(f"step {i+1:3d}/{steps}  {display}")

    return x


def _load_for_sampling(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"])
    model = DLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def _decode(ids, itos, mask_id):
    return "".join(itos[int(i)] if i != mask_id else "_" for i in ids)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/ckpt.pt")
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9,
                   help="nucleus truncation before categorical sampling (1.0 disables)")
    p.add_argument("--no-self-cond", action="store_true",
                   help="disable inference-time self-conditioning")
    p.add_argument("--ablate", action="store_true",
                   help="sweep steps in {1, 4, 16, 64, 256} to show the lesson")
    p.add_argument("--verbose", action="store_true",
                   help="print intermediate denoising states")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")

    model, cfg = _load_for_sampling(args.ckpt, device)
    data_dir = os.path.join(os.path.dirname(__file__), cfg.data_dir)
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        itos = pickle.load(f)["itos"]

    if args.ablate:
        print(f"=== ablation: length={args.length}, varying steps ===")
        for s in (1, 4, 16, 64, 256):
            out = generate(model, length=args.length, steps=s, device=device,
                           temperature=args.temperature, top_p=args.top_p,
                           self_cond=not args.no_self_cond)
            print(f"\n--- steps={s} ---")
            print(_decode(out[0].tolist(), itos, model.mask_id))
    else:
        out = generate(model, length=args.length, steps=args.steps,
                       device=device, temperature=args.temperature,
                       top_p=args.top_p, self_cond=not args.no_self_cond,
                       verbose=args.verbose, itos=itos)
        print("\n=== final ===")
        print(_decode(out[0].tolist(), itos, model.mask_id))
