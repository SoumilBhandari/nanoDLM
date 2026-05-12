"""Download tiny Shakespeare and tokenize it at the character level.

Output:
  data/train.bin, data/val.bin   (uint16 token ids)
  data/meta.pkl                  (vocab_size, stoi, itos)

The mask token is NOT in the vocab here — it's appended by the model so this
file stays byte-identical to nanoGPT's char-level prepare script.
"""
import os
import pickle
import urllib.request

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

input_path = os.path.join(DATA, "input.txt")
if not os.path.exists(input_path):
    print(f"downloading tiny shakespeare -> {input_path}")
    urllib.request.urlretrieve(URL, input_path)

with open(input_path) as f:
    text = f.read()
print(f"corpus length: {len(text):,} chars")

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
print(f"vocab size: {vocab_size}")

n = len(text)
train_text, val_text = text[: int(n * 0.9)], text[int(n * 0.9) :]
train_ids = np.array([stoi[c] for c in train_text], dtype=np.uint16)
val_ids = np.array([stoi[c] for c in val_text], dtype=np.uint16)
train_ids.tofile(os.path.join(DATA, "train.bin"))
val_ids.tofile(os.path.join(DATA, "val.bin"))
with open(os.path.join(DATA, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)

print(f"train: {len(train_ids):,} tokens   val: {len(val_ids):,} tokens")
