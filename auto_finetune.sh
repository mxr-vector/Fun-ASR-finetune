#!/bin/bash
# auto_finetune.sh - 一键运行全部三阶段训练
# Usage: nohup bash auto_finetune.sh > full_train.log 2>&1 &

set -e  # 遇错即停

# 错误处理函数
trap 'echo "❌ Stage $stage failed! Check logs."; exit 1' ERR

echo "========================================="
echo "Starting Full 3-Stage Training Pipeline"
echo "========================================="

# 循环执行三个阶段
for stage in 1 2 3; do
    stage_names=("Warmup" "Adaptation" "Fine-tuning")
    echo ""
    echo "[$stage/3] Starting Stage $stage: ${stage_names[$stage-1]}..."
    
    if bash finetune_stage.sh $stage; then
        echo "✓ Stage $stage completed successfully!"
    else
        echo "❌ Stage $stage failed!"
        echo "Check log: ./outputs/stage${stage}_*/log.txt"
        exit 1
    fi
done

echo ""
echo "========================================="
echo "✓ All 3 stages completed successfully!"
echo "========================================="
echo "Final model: ./outputs/stage3_finetune/model.pt.best"