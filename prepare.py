"""Download a corpus and tokenize it at the character level.

Two datasets:
  shakespeare (default): ~1 MB of tiny shakespeare from karpathy/char-rnn.
                         The canonical educational corpus.
  tinystories:           a capped slice of roneneldan/TinyStories from
                         HuggingFace. Purpose-built for tiny LMs to produce
                         coherent narratives — char-level Shakespeare is too
                         small and too archaic for the model to produce
                         readable modern English. Default cap is 50 MB
                         streamed from the start of the file, which gives
                         ~50K short stories. Pass --max-bytes to change.

Note on BPE: we deliberately stay char-level. Switching to GPT-2 BPE
(~50K vocab) would inflate the embedding table to ~19M params, making
the model ~30M params total and breaking the "tiny" identity. At this
scale char-level + a slightly larger sequence window is the right
trade.

Output:
  data/train.bin, data/val.bin   (uint16 token ids, 90/10 split)
  data/meta.pkl                  (vocab_size, stoi, itos, dataset name)

The mask token is NOT in the vocab — it's appended by the model.
"""
import argparse
import os
import pickle
import urllib.request

import numpy as np


SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-train.txt"
)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def download_shakespeare(path: str) -> None:
    print(f"downloading tiny shakespeare -> {path}")
    urllib.request.urlretrieve(SHAKESPEARE_URL, path)


def download_tinystories(path: str, max_bytes: int) -> None:
    """Stream the first `max_bytes` of TinyStoriesV2-GPT4-train.txt."""
    print(f"streaming TinyStories (first {max_bytes // 1024 // 1024} MB) -> {path}")
    req = urllib.request.Request(
        TINYSTORIES_URL,
        headers={"Range": f"bytes=0-{max_bytes - 1}"},
    )
    written = 0
    with urllib.request.urlopen(req) as response, open(path, "wb") as f:
        while written < max_bytes:
            chunk = response.read(min(1024 * 1024, max_bytes - written))
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
            if written % (10 * 1024 * 1024) == 0:
                print(f"  ...{written // 1024 // 1024} MB")
    # Trim back to the last complete story to avoid mid-story truncation
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")
    # TinyStories separates stories with the literal "<|endoftext|>" marker;
    # drop everything after the last one we have.
    last = text.rfind("<|endoftext|>")
    if last > 0:
        text = text[: last + len("<|endoftext|>")]
    # Strip the EOT markers entirely — char-level vocab can't include them as
    # a single token, and the model doesn't need them since each batch is a
    # random slice anyway.
    text = text.replace("<|endoftext|>", "\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  wrote {len(text):,} chars")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="shakespeare",
                   choices=["shakespeare", "tinystories"])
    p.add_argument("--max-bytes", type=int, default=50 * 1024 * 1024,
                   help="byte cap for streaming TinyStories (default 50 MB)")
    args = p.parse_args()

    os.makedirs(DATA, exist_ok=True)
    input_path = os.path.join(DATA, "input.txt")

    if not os.path.exists(input_path) or os.path.getsize(input_path) < 1000:
        if args.dataset == "shakespeare":
            download_shakespeare(input_path)
        else:
            download_tinystories(input_path, args.max_bytes)
    else:
        print(f"reusing existing {input_path} ({os.path.getsize(input_path):,} bytes)")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"corpus length: {len(text):,} chars")

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    print(f"vocab size: {vocab_size}")
    if vocab_size > 256:
        # uint16 still fits up to 65535; flag anyway because it's unusual.
        print(f"warning: vocab_size={vocab_size} is large for char-level "
              f"(unicode in the corpus?). Stored as uint16.")

    n = len(text)
    train_text, val_text = text[: int(n * 0.9)], text[int(n * 0.9):]
    train_ids = np.array([stoi[c] for c in train_text], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in val_text], dtype=np.uint16)
    train_ids.tofile(os.path.join(DATA, "train.bin"))
    val_ids.tofile(os.path.join(DATA, "val.bin"))
    with open(os.path.join(DATA, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos,
                     "dataset": args.dataset}, f)
    print(f"train: {len(train_ids):,} tokens   val: {len(val_ids):,} tokens")


if __name__ == "__main__":
    main()
