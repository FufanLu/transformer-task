"""Reload a submitted checkpoint the way a grader would and recompute validation perplexity.

Usage: python scripts/verify_checkpoint.py outputs/final [eval_batch_size]

Loads <dir>/config.yaml and <dir>/model.pt (mapped to CPU first), checks the file size,
then runs the provided evaluate() over the validation split.
"""

import os
import sys
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch
import wandb
from omegaconf import OmegaConf

from lm.model import DecoderLM
from lm.train import evaluate, sequential_batch_sampler
from lm.utils import determine_device


def main():
    model_dir = sys.argv[1]
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model_path = os.path.join(model_dir, "model.pt")
    print(f"model.pt: {os.path.getsize(model_path) / 1e6:.1f} MB")

    wandb.init(mode="disabled")
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device()
    model = DecoderLM(tokenizer.n_vocab, **config.model_config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device).eval()

    val_tokens = torch.from_numpy(np.load("data/tokens.npz")["val"].astype(int))
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else config.batch_size
    sampler = sequential_batch_sampler(val_tokens, device, batch_size, config.seq_len)
    print("recomputed:", evaluate(model, sampler, nullcontext()))


if __name__ == "__main__":
    main()
