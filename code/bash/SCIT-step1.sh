#!/usr/bin/env bash
set -euo pipefail

# Load environment variables (must be in same dir or specify full path)
source "$(dirname "$0")/env.sh"

echo "Using bucket: $S3_BUCKET_URI"
echo "Sending alerts to: $ALERT_EMAIL_TO"

unset WORLD_SIZE LOCAL_RANK RANK
unset CUDA_VISIBLE_DEVICES

torchrun /root/RAHF/code/step1/SCIT-step1.py \
    --base_model  "Liuwenhao2022/RAHF-SFT" \
    --seed 42 \
    --data_path "/root/RAHF/data/ultrafeedback/rm" \
    --batch_size 64 \
    --micro_batch_size 1 \
    --num_epochs 2 \
    --learning_rate 2e-5 \
    --max_length 768 \
    --warmup_ratio 0.1 \
    --save_steps 400 \
    --output_dir '/root/RAHF/model/SCIT/hir' \
    --resume_from_checkpoint '/root/RAHF/model/SCIT/hir/checkpoint-400'

