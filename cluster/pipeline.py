"""Q3.4 round 2 + Q3.5 final training, decided by rules fixed before seeing the results.

1. Round 1 (A, B, C: same d256/L8 model, different training recipes) runs in another
   job; wait for its eval.json files and train any config that is still missing.
2. Round 2 reuses the training recipe of the best round-1 config (lowest validation
   perplexity) on two other model shapes, each within the 1e15 FLOP budget:
     D = d256 / 12 layers / 8 heads (deeper), E = d320 / 6 layers / 5 heads (wider).
3. The best of A-E fixes the model hyperparameters; the final model keeps them and is
   trained with ~1e17 FLOPs: batch 256 x seq_len, the winning max/min lr, 2% warmup.

Run from the A2 directory: python cluster/pipeline.py
"""

import json
import os
import subprocess
import sys
import time

from omegaconf import OmegaConf

sys.path.insert(0, "src")
from lm.model import DecoderLM  # noqa: E402

N_VOCAB = 50257
ROUND1 = ["A_d256L8_lr5e-4", "B_d256L8_lr2e-3", "C_d256L8_lr2e-3_bs16"]
ROUND2_MODELS = {
    "D_d256L12": dict(n_embd=256, n_head=8, n_positions=128, n_layer=12),
    "E_d320L6": dict(n_embd=320, n_head=5, n_positions=128, n_layer=6),
}
WAIT_FOR_ROUND1_S = 45 * 60


def eval_path(name: str) -> str:
    return f"outputs/hparam/{name}/eval.json"


def val_ppl(name: str) -> float:
    with open(eval_path(name)) as f:
        return json.load(f)["val-perplexity"]


def train(config_path: str) -> None:
    print(f"\n### training {config_path}", flush=True)
    subprocess.run([sys.executable, "src/lm/train.py", config_path], check=True)


def steps_for_budget(model_config: dict, batch_size: int, seq_len: int, accum: int, budget: float) -> int:
    flops_per_token = DecoderLM(N_VOCAB, **model_config).flops_per_token
    return int(budget // (flops_per_token * batch_size * seq_len * accum))


def write_config(path: str, base, **overrides) -> str:
    cfg = OmegaConf.merge(base, overrides)
    OmegaConf.save(cfg, path)
    print(f"--- {path}\n{OmegaConf.to_yaml(cfg)}", flush=True)
    return path


def main():
    # 1. round 1: produced by the V100 job; fill in anything missing after a grace period
    deadline = time.time() + WAIT_FOR_ROUND1_S
    while not all(os.path.exists(eval_path(n)) for n in ROUND1) and time.time() < deadline:
        time.sleep(30)
    for name in ROUND1:
        if not os.path.exists(eval_path(name)):
            train(f"configs/hparam/{name}.yaml")

    best1 = min(ROUND1, key=val_ppl)
    print("round 1:", {n: round(val_ppl(n), 2) for n in ROUND1}, "-> best recipe", best1, flush=True)
    recipe = OmegaConf.load(f"configs/hparam/{best1}.yaml")

    # 2. round 2: winning recipe on a deeper and a wider model, each <= 1e15 FLOPs
    for name, model_config in ROUND2_MODELS.items():
        if os.path.exists(eval_path(name)):
            continue
        steps = steps_for_budget(model_config, recipe.batch_size, recipe.seq_len, recipe.grad_accumulation_steps, 1e15)
        warmup = round(recipe.num_warmup_steps * steps / recipe.num_training_steps)
        path = write_config(
            f"configs/hparam/{name}.yaml",
            recipe,
            output_dir=f"outputs/hparam/{name}",
            model_config=model_config,
            num_training_steps=steps,
            num_warmup_steps=warmup,
        )
        train(path)

    everything = ROUND1 + list(ROUND2_MODELS)
    best = min(everything, key=val_ppl)
    print("all configs:", {n: round(val_ppl(n), 2) for n in everything}, "-> best", best, flush=True)

    # 3. final model: same model hyperparameters, ~1e17 FLOPs
    best_cfg = OmegaConf.load(f"configs/hparam/{best}.yaml")
    model_config = OmegaConf.to_container(best_cfg.model_config)
    batch_size, seq_len = 256, best_cfg.seq_len
    steps = steps_for_budget(model_config, batch_size, seq_len, 1, 1e17)
    path = write_config(
        "configs/final.yaml",
        best_cfg,
        output_dir="outputs/final",
        batch_size=batch_size,
        grad_accumulation_steps=1,
        num_training_steps=steps,
        num_warmup_steps=max(1, steps // 50),
    )
    train(path)
    with open("outputs/final/eval.json") as f:
        print("FINAL:", f.read(), flush=True)


if __name__ == "__main__":
    main()
