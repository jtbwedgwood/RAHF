#!/bin/zsh
#SBATCH --job-name=dual-step1
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1           # single GPU is typically enough for 7B + LoRA
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/jwedgwoo/RAHF/code
#SBATCH --output=/home/jwedgwoo/RAHF/code/logs/%x-%j.out
#SBATCH --error=/home/jwedgwoo/RAHF/code/logs/%x-%j.err

set -euo pipefail

# huggingface tokens
export HF_HOME=/home/jwedgwoo/RAHF/.cache/huggingface
export HF_TOKEN=$(cat /home/jwedgwoo/RAHF/huggingface_token.txt)
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

# Activate venv from the actual filesystem, not /var/spool
source /home/jwedgwoo/RAHF/.venv/bin/activate

# Slurm provides CUDA-visible devices; don't override them
# WORLD_SIZE=1 implies single-process training
export WORLD_SIZE=1

# Run the two training passes explicitly (avoids overriding CUDA_VISIBLE_DEVICES)
python step1/DUAL-step1.py \
  --model_path "Liuwenhao2022/RAHF-SFT" \
  --data_path  "../data/ultrafeedback/rm" \
  --output_dir "../model/DUAL/good" \
  --preference_type "chosen"

python step1/DUAL-step1.py \
  --model_path "Liuwenhao2022/RAHF-SFT" \
  --data_path  "../data/ultrafeedback/rm" \
  --output_dir "../model/DUAL/bad" \
  --preference_type "rejected"