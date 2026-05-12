from dataclasses import dataclass


@dataclass
class Config:
    # data
    data_dir: str = "data"
    block_size: int = 256
    vocab_size: int = 65          # set by prepare.py; +1 for [MASK] is added inside the model

    # model (~10M params at these settings on char-level)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False

    # optim
    batch_size: int = 64
    max_steps: int = 5000
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100

    # eval / logging
    eval_interval: int = 250
    eval_iters: int = 50
    sample_interval: int = 500

    # diffusion
    eps: float = 1e-3             # min mask ratio to avoid 1/t blowup

    # system
    device: str = "cuda"          # falls back to cpu/mps in train.py
    dtype: str = "bfloat16"       # autocast dtype on cuda
    compile: bool = True
    seed: int = 1337
    out_dir: str = "out"
