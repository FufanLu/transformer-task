"""Plot train-loss curves from offline wandb runs.

Usage: python scripts/plot_curves.py OUT.png LABEL=wandb/offline-run-XXX [LABEL=...] [--lr]
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wandb_history import read_history


def main():
    out, specs = sys.argv[1], [a for a in sys.argv[2:] if "=" in a]
    show_lr = "--lr" in sys.argv
    fig, ax = plt.subplots(figsize=(7, 4))
    for spec in specs:
        label, run_dir = spec.split("=", 1)
        rows = [r for r in read_history(run_dir) if "train-loss" in r]
        steps = [r["_step"] for r in rows]
        ax.plot(steps, [r["train-loss"] for r in rows], label=label, linewidth=1.2)
        if show_lr:
            ax2 = ax.twinx()
            ax2.plot(steps, [r["learning-rate"] for r in rows], color="gray", linestyle="--", linewidth=1)
            ax2.set_ylabel("learning rate", color="gray")
    ax.set_xlabel("step")
    ax.set_ylabel("train loss (mean of last 20 steps)")
    ax.grid(alpha=0.3)
    if len(specs) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()
