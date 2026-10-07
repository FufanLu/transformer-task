#!/bin/bash
#SBATCH --job-name=hw2-final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/%u/llms-hw2/logs/%x_%j.out
# Q3.4 round 2 + Q3.5 final run (see cluster/pipeline.py for the decision rules).
set -euo pipefail
ROOT=/scratch/$USER/llms-hw2
cd "$ROOT/A2"
module load anaconda3/2024.06
source activate "$ROOT/env"
export PYTHONPATH=$ROOT/A2/src
export TIKTOKEN_CACHE_DIR=$ROOT/tiktoken
export WANDB_MODE=offline WANDB_DIR=$ROOT
export TQDM_MININTERVAL=20
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python cluster/pipeline.py
echo "=== pipeline done $(date)"
