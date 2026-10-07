"""Perplexity of generated continuations under Pythia (continuation tokens only).

Usage: python scripts/score_continuations.py outputs/q4_2/*/generation*.jsonl

Each input line is {"prefix": ..., "generation": ...}, as written by generate.py and
use_pythia.py. The prefix is fed as context and only the continuation tokens are
scored, which matches how generate.py reports perplexity for our own model.
"""

import argparse
import json
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm.utils import determine_device


@torch.inference_mode()
def continuation_nll(model, tokenizer, device, prefix: str, generation: str) -> tuple[float, int]:
    n_prefix = len(tokenizer(prefix)["input_ids"])
    ids = tokenizer(prefix + generation, return_tensors="pt")["input_ids"].to(device)
    if ids.shape[1] <= n_prefix:
        return 0.0, 0
    log_probs = model(input_ids=ids).logits[0, :-1].float().log_softmax(dim=-1)
    targets = ids[0, 1:]
    token_lp = log_probs.gather(1, targets[:, None])[:, 0]
    # position i predicts token i + 1, so continuation tokens start at n_prefix - 1
    cont_lp = token_lp[n_prefix - 1 :]
    return -cont_lp.sum().item(), cont_lp.numel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--model", default="EleutherAI/pythia-1.4b")
    args = parser.parse_args()

    device = determine_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    for path in args.files:
        total_nll, total_tokens = 0.0, 0
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                nll, n = continuation_nll(model, tokenizer, device, row["prefix"], row["generation"])
                total_nll, total_tokens = total_nll + nll, total_tokens + n
        print(f"{path}\tPythia PPL = {math.exp(total_nll / total_tokens):.2f}\t({total_tokens} tokens)")


if __name__ == "__main__":
    main()
