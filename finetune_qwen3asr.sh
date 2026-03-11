#!/usr/bin/env bash
# =============================================================================
# 三阶段渐进式训练脚本 - Qwen3-ASR-1.7B
# Stage 1: 通用/专业 50/50 混合热身
# Stage 2: 通用/专业 20/80 过渡精调
# Stage 3: 纯专业数据终训
# =============================================================================
set -euo pipefail

workspace=$(pwd)

export CUDA_VISIBLE_DEVICES="0,1"
gpu_num=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F "," '{print NF}')

# ─── 路径配置 ────────────────────────────────────────────────────────────────
MODEL_PATH="models/Qwen3-ASR-1.7B"
DATA_ROOT="./data/staged"
OUTPUT_ROOT="./output/staged"

# ─── 各阶段最优 checkpoint 路径（供下一阶段加载） ───────────────────────────
STAGE1_CKPT="${OUTPUT_ROOT}/stage1/best_model"
STAGE2_CKPT="${OUTPUT_ROOT}/stage2/best_model"

# ─── 分布式参数（数组形式，避免多行字符串展开错误） ──────────────────────────
DISTRIBUTED_ARGS=(
    --nnodes        "${WORLD_SIZE:-1}"
    --nproc_per_node "${gpu_num}"
    --node_rank     "${RANK:-0}"
    --master_addr   "${MASTER_ADDR:-127.0.0.1}"
    --master_port   "${MASTER_PORT:-26668}"
)

# ─── 工具函数 ─────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

train_run() {
    local stage_data="" stage_out="" input_model=""
    local batch_size="" grad_acc="" lr="" epochs="" save_steps=""
    local warmup_ratio="0.05"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --stage_data)    stage_data="$2";    shift 2 ;;
            --stage_out)     stage_out="$2";     shift 2 ;;
            --input_model)   input_model="$2";   shift 2 ;;
            --batch_size)    batch_size="$2";    shift 2 ;;
            --grad_acc)      grad_acc="$2";      shift 2 ;;
            --lr)            lr="$2";            shift 2 ;;
            --epochs)        epochs="$2";        shift 2 ;;
            --save_steps)    save_steps="$2";    shift 2 ;;
            --warmup_ratio)  warmup_ratio="$2";  shift 2 ;;
            *) log "未知参数: $1"; exit 1 ;;
        esac
    done

    mkdir -p "${stage_out}"
    local log_file="${stage_out}/log.txt"

    # ── 阶段内断点续传：检测本阶段 output_dir 内是否有中断的 step checkpoint ──
    local resume_ckpt=""
    resume_ckpt=$(ls -td "${stage_out}"/checkpoint-* 2>/dev/null | head -1 || true)

    log "════════════════════════════════════════════"
    log "  输入模型 : ${input_model}"
    log "  断点续传 : ${resume_ckpt:-无}"
    log "  训练数据 : ${stage_data}/train.jsonl"
    log "  验证数据 : ${stage_data}/val.jsonl"
    log "  输出目录 : ${stage_out}"
    log "════════════════════════════════════════════"

    if [ -n "${resume_ckpt}" ]; then
        log ">> 检测到中断 checkpoint: ${resume_ckpt}，从断点续训"
    else
        log ">> 未发现断点，从头开始训练"
    fi

    # resume_ckpt 非空时追加 --resume_from_checkpoint，否则该行不展开
    torchrun "${DISTRIBUTED_ARGS[@]}" \
        qwen3_asr_sft.py \
            --model_path            "${input_model}" \
            ${resume_ckpt:+--resume_from_checkpoint "${resume_ckpt}"} \
            --train_file            "${stage_data}/train.jsonl" \
            --eval_file             "${stage_data}/val.jsonl" \
            --output_dir            "${stage_out}" \
            --batch_size            "${batch_size}" \
            --grad_acc              "${grad_acc}" \
            --lr                    "${lr}" \
            --epochs                "${epochs}" \
            --warmup_ratio          "${warmup_ratio}" \
            --log_steps             10 \
            --save_strategy         steps \
            --save_steps            "${save_steps}" \
            --save_total_limit      3 \
            --num_workers           4 \
            --pin_memory            1 \
            --persistent_workers    1 \
            --prefetch_factor       2 \
        &> "${log_file}"

    if [ $? -eq 0 ]; then
        log "✓ 训练完成 → ${stage_out}"
    else
        log "✗ 训练失败，日志: ${log_file}"
        exit 1
    fi
}

# ─── 解析命令行参数（允许从指定阶段续跑） ────────────────────────────────────
START_STAGE=${1:-1}
log "从 Stage ${START_STAGE} 开始训练"

# ─── Stage 1: 50/50 热身（直接加载预训练权重） ───────────────────────────────
if [ "${START_STAGE}" -le 1 ]; then
    log "=== Stage 1: 通用/专业 50-50 混合热身 ==="
    # 有效 BS = batch_size(32) × grad_acc(4) × GPU数(2) = 256
    train_run \
        --stage_data   "${DATA_ROOT}/stage1" \
        --stage_out    "${OUTPUT_ROOT}/stage1" \
        --input_model  "${MODEL_PATH}" \
        --batch_size   32 \
        --grad_acc     4 \
        --lr           2e-5 \
        --epochs       1 \
        --save_steps   200 \
        --warmup_ratio 0.05
fi

# ─── Stage 2: 20/80 过渡（加载 Stage 1 checkpoint） ─────────────────────────
if [ "${START_STAGE}" -le 2 ]; then
    log "=== Stage 2: 通用/专业 20-80 过渡精调 ==="
    [ -d "${STAGE1_CKPT}" ] || { log "错误: 找不到 Stage 1 checkpoint: ${STAGE1_CKPT}"; exit 1; }
    # 有效 BS = batch_size(16) × grad_acc(8) × GPU数(2) = 256
    train_run \
        --stage_data   "${DATA_ROOT}/stage2" \
        --stage_out    "${OUTPUT_ROOT}/stage2" \
        --input_model  "${STAGE1_CKPT}" \
        --batch_size   16 \
        --grad_acc     8 \
        --lr           8e-6 \
        --epochs       2 \
        --save_steps   100 \
        --warmup_ratio 0.03
fi

# ─── Stage 3: 纯专业精调（加载 Stage 2 checkpoint） ─────────────────────────
if [ "${START_STAGE}" -le 3 ]; then
    log "=== Stage 3: 纯专业数据终训 ==="
    [ -d "${STAGE2_CKPT}" ] || { log "错误: 找不到 Stage 2 checkpoint: ${STAGE2_CKPT}"; exit 1; }
    # 有效 BS = batch_size(8) × grad_acc(16) × GPU数(2) = 256
    train_run \
        --stage_data   "${DATA_ROOT}/stage3" \
        --stage_out    "${OUTPUT_ROOT}/stage3" \
        --input_model  "${STAGE2_CKPT}" \
        --batch_size   8 \
        --grad_acc     16 \
        --lr           2e-6 \
        --epochs       3 \
        --save_steps   50 \
        --warmup_ratio 0.02
fi

# ─── 完成 ────────────────────────────────────────────────────────────────────
log "所有阶段训练完成！"
log "最终模型位于: ${OUTPUT_ROOT}/stage3"