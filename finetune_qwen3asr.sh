#!/usr/bin/env bash
# =============================================================================
# 三阶段渐进式训练脚本 - Qwen3-ASR-1.7B (LoRA 微调版)
# Stage 1: 通用/专业 50/50 混合热身
# Stage 2: 通用/专业 20/80 过渡精调
# Stage 3: 纯专业数据终训
# =============================================================================
set -euo pipefail

workspace=$(pwd)

export CUDA_VISIBLE_DEVICES="2,3"
gpu_num=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F "," '{print NF}')

# ─── 显存优化：避免碎片化 ─────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── 路径配置 ────────────────────────────────────────────────────────────────
MODEL_PATH="models/Qwen3-ASR-1.7B"
DATA_ROOT="./data/staged"
OUTPUT_ROOT="./outputs/staged"

# ─── 各阶段最优 checkpoint 路径（供下一阶段加载） ───────────────────────────
STAGE1_CKPT="${OUTPUT_ROOT}/stage1/best_model"
STAGE2_CKPT="${OUTPUT_ROOT}/stage2/best_model"

# ─── LoRA 配置 ────────────────────────────────────────────────────────────────
USE_LORA=1
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"

# ─── Gradient Checkpointing ──────────────────────────────────────────────────
GRADIENT_CHECKPOINTING=1

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
    local lora_adapter_path=""
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
            --lora_adapter_path) lora_adapter_path="$2"; shift 2 ;;
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
    log "  LoRA     : USE_LORA=${USE_LORA}, r=${LORA_R}, alpha=${LORA_ALPHA}"
    log "  GradCkpt : ${GRADIENT_CHECKPOINTING}"
    log "  断点续传 : ${resume_ckpt:-无}"
    log "  LoRA续训 : ${lora_adapter_path:-无}"
    log "  训练数据 : ${stage_data}/train.jsonl"
    log "  验证数据 : ${stage_data}/val.jsonl"
    log "  输出目录 : ${stage_out}"
    log "  BS/GPU   : ${batch_size}, GradAcc: ${grad_acc}"
    log "  有效BS   : $((batch_size * grad_acc * gpu_num))"
    log "════════════════════════════════════════════"

    if [ -n "${resume_ckpt}" ]; then
        log ">> 检测到中断 checkpoint: ${resume_ckpt}，从断点续训"
    else
        log ">> 未发现断点，从头开始训练"
    fi

    # resume_ckpt 非空时追加 --resume_from，否则该行不展开
    # 使用 || true 避免 set -e 导致脚本直接退出，以便捕获退出码
    local exit_code=0
    torchrun "${DISTRIBUTED_ARGS[@]}" \
        tools/qwen3_asr_sft.py \
            --model_path            "${input_model}" \
            ${resume_ckpt:+--resume_from "${resume_ckpt}"} \
            ${lora_adapter_path:+--lora_adapter_path "${lora_adapter_path}"} \
            --train_file            "${stage_data}/train_qwen3asr.jsonl" \
            --eval_file             "${stage_data}/val_qwen3asr.jsonl" \
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
            --use_lora              "${USE_LORA}" \
            --lora_r                "${LORA_R}" \
            --lora_alpha            "${LORA_ALPHA}" \
            --lora_dropout          "${LORA_DROPOUT}" \
            --lora_target_modules   "${LORA_TARGET_MODULES}" \
            --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
        &> "${log_file}" || exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        log "✓ 训练完成 → ${stage_out}"
    else
        log "✗ 训练失败（exit code: ${exit_code}），日志: ${log_file}"
        exit 1
    fi
}

# ─── 解析命令行参数（允许从指定阶段续跑） ────────────────────────────────────
START_STAGE=${1:-1}
log "从 Stage ${START_STAGE} 开始训练"

# ─── Stage 1: 50/50 热身（直接加载预训练权重） ───────────────────────────────
if [ "${START_STAGE}" -le 1 ]; then
    log "=== Stage 1: 通用/专业 50-50 混合热身 ==="
    # 有效 BS = batch_size(4) × grad_acc(8) × GPU数(2) = 64
    train_run \
        --stage_data   "${DATA_ROOT}/stage1" \
        --stage_out    "${OUTPUT_ROOT}/stage1" \
        --input_model  "${MODEL_PATH}" \
        --batch_size   4 \
        --grad_acc     8 \
        --lr           1e-4 \
        --epochs       3 \
        --save_steps   400 \
        --warmup_ratio 0.03
fi

# ─── Stage 2: 20/80 过渡（加载 Stage 1 checkpoint） ─────────────────────────
if [ "${START_STAGE}" -le 2 ]; then
    log "=== Stage 2: 通用/专业 20-80 过渡精调 ==="
    [ -d "${STAGE1_CKPT}/adapter" ] || { log "错误: 找不到 Stage 1 adapter: ${STAGE1_CKPT}/adapter"; exit 1; }
    # 有效 BS = batch_size(4) × grad_acc(8) × GPU数(2) = 64
    train_run \
        --stage_data   "${DATA_ROOT}/stage2" \
        --stage_out    "${OUTPUT_ROOT}/stage2" \
        --input_model  "${MODEL_PATH}" \
        --lora_adapter_path "${STAGE1_CKPT}/adapter" \
        --batch_size   4 \
        --grad_acc     8 \
        --lr           5e-5 \
        --epochs       4 \
        --save_steps   200 \
        --warmup_ratio 0.03
fi

# ─── Stage 3: 纯专业精调（加载 Stage 2 checkpoint） ─────────────────────────
if [ "${START_STAGE}" -le 3 ]; then
    log "=== Stage 3: 纯专业数据终训 ==="
    [ -d "${STAGE2_CKPT}/adapter" ] || { log "错误: 找不到 Stage 2 adapter: ${STAGE2_CKPT}/adapter"; exit 1; }
    # 有效 BS = batch_size(4) × grad_acc(8) × GPU数(2) = 64
    train_run \
        --stage_data   "${DATA_ROOT}/stage3" \
        --stage_out    "${OUTPUT_ROOT}/stage3" \
        --input_model  "${MODEL_PATH}" \
        --lora_adapter_path "${STAGE2_CKPT}/adapter" \
        --batch_size   4 \
        --grad_acc     8 \
        --lr           3e-5 \
        --epochs       5 \
        --save_steps   100 \
        --warmup_ratio 0.02
fi

# ─── 完成 ────────────────────────────────────────────────────────────────────
log "所有阶段训练完成！"
log "最终模型位于: ${OUTPUT_ROOT}/stage3"
log ""
log "如需将 LoRA 合并回完整模型，请运行:"
log "  python tools/lora_merge_qwen3asr.py \\"
log "      --base_model_path ${MODEL_PATH} \\"
log "      --adapter_path ${OUTPUT_ROOT}/stage3/best_model/adapter \\"
log "      --output_path ${OUTPUT_ROOT}/merged_model"