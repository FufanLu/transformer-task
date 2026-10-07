#!/bin/bash
#SBATCH --job-name=hw2-setup
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/llms-hw2/logs/setup_%j.out
# Builds the training env on /scratch (purged after 28 days; rerun this to rebuild).
# Runs as a job because the login node kills heavy processes such as `import torch`.
set -euo pipefail
ROOT=/scratch/$USER/llms-hw2
module load anaconda3/2024.06
conda create -y -p "$ROOT/env" python=3.11 >/dev/null
source activate "$ROOT/env"
# cu121 wheels run on every GPU type here (P100 through H200)
pip install -q --no-input torch --index-url https://download.pytorch.org/whl/cu121
pip install -q --no-input einops numpy omegaconf tiktoken tqdm wandb rich pytest
# compute nodes may have no internet: cache the GPT-2 BPE files now
export TIKTOKEN_CACHE_DIR=$ROOT/tiktoken
python -c "import tiktoken; print('gpt2 vocab', tiktoken.get_encoding('gpt2').n_vocab)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "setup done"
