#!/bin/bash
# finetune_stage.sh - 一键运行全部三阶段训练
# nohup bash auto_finuetune.sh > full_train.log 2>&1 &

set -e  # 任何错误立即停止

echo "========================================="
echo "Starting Full 3-Stage Training Pipeline"
echo "========================================="

# 阶段1
echo ""
echo "[1/3] Starting Stage 1: Warmup..."
bash finetune_stage.sh 1
if [ $? -ne 0 ]; then
    echo "Stage 1 failed!"
    exit 1
fi

# 阶段2
echo ""
echo "[2/3] Starting Stage 2: Adaptation..."
bash finetune_stage.sh 2
if [ $? -ne 0 ]; then
    echo "Stage 2 failed!"
    exit 1
fi

# 阶段3
echo ""
echo "[3/3] Starting Stage 3: Fine-tuning..."
bash finetune_stage.sh 3
if [ $? -ne 0 ]; then
    echo "Stage 3 failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ All 3 stages completed successfully!"
echo "========================================="
echo "Final model: ./outputs/stage3_finetune/valid.acc.best.pth"