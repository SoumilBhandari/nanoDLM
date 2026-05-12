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


@torch.no_grad()
def generate(model, length=256, steps=64, temperature=1.0, device="cpu", verbose=False, itos=None):
    """Generate `length` tokens via `steps` rounds of low-confidence remasking."""
    model.eval()
    x = torch.full((1, length), model.mask_id, dtype=torch.long, device=device)

    for i in range(steps):
        logits = model(x) / temperature
        probs = logits.softmax(-1)
        conf, pred = probs.max(-1)                                    # (1, L)

        # Schedule: how many tokens should remain masked AFTER this step.
        frac_remaining = 1.0 - (i + 1) / steps
        n_keep_masked = int(frac_remaining * length)

        is_masked = (x == model.mask_id)
        # Only consider currently-masked positions for unmasking.
        conf_masked = conf.masked_fill(~is_masked, float("-inf"))

        if n_keep_masked == 0:
            unmask = is_masked
        else:
            # Threshold = lowest confidence we will KEEP unmasked-this-round.
            kth = conf_masked.kthvalue(n_keep_masked + 1, dim=-1).values
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
                           temperature=args.temperature)
            print(f"\n--- steps={s} ---")
            print(_decode(out[0].tolist(), itos, model.mask_id))
    else:
        out = generate(model, length=args.length, steps=args.steps,
                       device=device, temperature=args.temperature,
                       verbose=args.verbose, itos=itos)
        print("\n=== final ===")
        print(_decode(out[0].tolist(), itos, model.mask_id))
