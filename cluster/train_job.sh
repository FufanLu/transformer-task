#!/bin/bash
#SBATCH --job-name=hw2-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/llms-hw2/logs/%x_%j.out
# Usage: sbatch [--job-name=NAME] [--time=HH:MM:SS] cluster/train_job.sh CONFIG [CONFIG ...]
# Trains each config in turn (paths relative to the A2 directory).
set -euo pipefail
ROOT=/scratch/$USER/llms-hw2
cd "$ROOT/A2"
module load anaconda3/2024.06
source activate "$ROOT/env"
export PYTHONPATH=$ROOT/A2/src
export TIKTOKEN_CACHE_DIR=$ROOT/tiktoken
export WANDB_MODE=offline WANDB_DIR=$ROOT
export TQDM_MININTERVAL=20   # keep the log readable
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if [ "${RUN_TESTS:-0}" = 1 ]; then
  python -m pytest -q -p no:cacheprovider tests/test_model.py tests/test_train.py tests/test_generate.py
fi
for cfg in "$@"; do
  echo "=== $cfg  $(date)"
  python src/lm/train.py "$cfg"
done
echo "=== all done $(date)"
