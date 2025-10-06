#!/usr/bin/env bash
set -euo pipefail

# Load environment variables (must be in same dir or specify full path)
source "$(dirname "$0")/env.sh"

echo "Using bucket: $S3_BUCKET_URI"
echo "Sending alerts to: $ALERT_EMAIL_TO"

unset WORLD_SIZE LOCAL_RANK RANK
unset CUDA_VISIBLE_DEVICES

# Log file
LOG_DIR=/workspace/ckpts/hir
LOG_FILE="$LOG_DIR/train.stdout.log"
mkdir -p "$LOG_DIR"
trap 'echo "$(date) [runner] caught SIGTERM" | tee -a $LOG_FILE' TERM

# Line-buffer app stdout/stderr so logs stream immediately, then tee to file
stdbuf -oL -eL torchrun --nproc_per_node=1 /root/RAHF/code/step1/SCIT-step1.py \
  --base_model "Liuwenhao2022/RAHF-SFT" \
  --seed 42 \
  --data_path "/root/RAHF/data/ultrafeedback/rm" \
  --batch_size 64 \
  --micro_batch_size 1 \
  --num_epochs 2 \
  --learning_rate 2e-5 \
  --max_length 768 \
  --warmup_ratio 0.1 \
  --save_steps 200 \
  --output_dir /workspace/ckpts/hir \
  --resume_from_checkpoint "/workspace/ckpts/hir/checkpoint-400" \
  2>&1 | tee -a "$LOG_FILE"

