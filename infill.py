"""Infilling demo: fix a prefix and a suffix, let the diffusion LM fill the middle.

This is the demo an autoregressive LM cannot do. nanoGPT can only continue
from a prefix because its attention is causal — it cannot condition on
tokens to the right of the current position. A masked-diffusion LM has
bidirectional attention by construction, so the same model that does
unconditional generation in `sample.py` does middle-completion here for
free: we just freeze the prefix and suffix positions and let the
low-confidence remasking loop denoise the middle.

Examples:
    # Shakespeare-trained model
    python infill.py --prefix "ROMEO:" --suffix "JULIET:" --middle-length 120

    # TinyStories-trained model
    python infill.py --prefix "Once upon a time" --suffix "happily ever after." --middle-length 80
"""
import argparse
import os
import pickle

import torch

from sample import _load_for_sampling, _decode, generate


def encode(text, stoi):
    try:
        return [stoi[c] for c in text]
    except KeyError as e:
        raise SystemExit(f"char {e!r} is not in the vocabulary built by prepare.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/ckpt.pt")
    p.add_argument("--prefix", required=True, help="text the passage must start with")
    p.add_argument("--suffix", required=True, help="text the passage must end with")
    p.add_argument("--middle-length", type=int, default=64,
                   help="how many tokens to generate between prefix and suffix")
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9,
                   help="nucleus truncation before categorical sampling (1.0 disables)")
    p.add_argument("--no-self-cond", action="store_true",
                   help="disable inference-time self-conditioning")
    p.add_argument("--verbose", action="store_true",
                   help="print intermediate denoising states (watch text crystallize)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")

    model, cfg = _load_for_sampling(args.ckpt, device)
    data_dir = os.path.join(os.path.dirname(__file__), cfg.data_dir)
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]

    prefix_ids = encode(args.prefix, stoi)
    suffix_ids = encode(args.suffix, stoi)
    total_length = len(prefix_ids) + args.middle_length + len(suffix_ids)
    if total_length > cfg.block_size:
        raise SystemExit(
            f"total length {total_length} exceeds block_size {cfg.block_size}; "
            f"shorten --middle-length, --prefix, or --suffix")

    x_init = torch.full((1, total_length), model.mask_id, dtype=torch.long, device=device)
    x_init[0, :len(prefix_ids)] = torch.tensor(prefix_ids, device=device)
    x_init[0, total_length - len(suffix_ids):] = torch.tensor(suffix_ids, device=device)

    out = generate(model, x_init=x_init, steps=args.steps,
                   temperature=args.temperature, top_p=args.top_p,
                   self_cond=not args.no_self_cond, device=device,
                   verbose=args.verbose, itos=itos)

    text = _decode(out[0].tolist(), itos, model.mask_id)
    print("\n=== final ===")
    print(text)
    print(f"\n(prefix:  {args.prefix!r}\n suffix:  {args.suffix!r}\n middle:  "
          f"{text[len(args.prefix):len(text) - len(args.suffix)]!r})")
