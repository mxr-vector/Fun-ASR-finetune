#!/bin/bash
# finetune_three_stage.sh - 三阶段训练，参数全部写死
# funASR paramforer系列模型使用训练脚本

workspace=`pwd`

export CUDA_VISIBLE_DEVICES="1,2,3"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

model_name_or_model_dir="/workspace/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

# 数据目录，分阶段
data_dir="/workspace/FunASR/data/staged"

# checkpoint 保存路径
stage1_ckpt="./outputs/stage1_warmup/model.pt.best"
stage2_ckpt="./outputs/stage2_adaptation/model.pt.best"

DISTRIBUTED_ARGS="
    --nnodes ${WORLD_SIZE:-1} \
    --nproc_per_node $gpu_num \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-26669}
"

train_run(){
    STAGE_NAME=$1
    TRAIN_DATA=$2
    VAL_DATA=$3
    MAX_EPOCH=$4
    LR=$5
    OUTPUT_DIR=$6
    MODEL_DIR=$7

    mkdir -p ${OUTPUT_DIR}
    log_file="${OUTPUT_DIR}/log.txt"
    echo "=============================="
    echo "Stage ${STAGE_NAME} Training"
    echo "Train data: ${TRAIN_DATA}"
    echo "Valid data: ${VAL_DATA}"
    echo "Output dir: ${OUTPUT_DIR}"
    echo "Learning rate: ${LR}"
    echo "=============================="

    # 判定续训
    CURRENT_STAGE_CKPT="${OUTPUT_DIR}/model.pt"
    if [ -f "${CURRENT_STAGE_CKPT}" ]; then
        echo ">> Found checkpoint: ${CURRENT_STAGE_CKPT}, resume training"
        MODEL_DIR="++model=${CURRENT_STAGE_CKPT}"
        RESUME_PARAM="++train_conf.resume=true"
    else
        echo ">> No checkpoint, init from: ${MODEL_DIR}"
        RESUME_PARAM="++train_conf.resume=false"
    fi

    deepspeed_config=/workspace/FunASR/example/deepspeed_conf/ds_stage1.json

    torchrun $DISTRIBUTED_ARGS \
    /workspace/FunASR/funasr/bin/train_ds.py \
    ${MODEL_DIR} \
    ${RESUME_PARAM} \
    ++trust_remote_code=true \
    ++train_data_set_list="${TRAIN_DATA}" \
    ++valid_data_set_list="${VAL_DATA}" \
    ++dataset_conf.data_split_num=1 \
    ++dataset_conf.batch_sampler="BatchSampler" \
    ++dataset_conf.batch_type="token" \
    ++dataset_conf.batch_size=8000 \
    ++dataset_conf.sort_size=1024 \
    ++dataset_conf.num_workers=4 \
    ++dataset_conf.shuffle=true \
    ++train_conf.max_epoch=${MAX_EPOCH} \
    ++train_conf.log_interval=1 \
    ++train_conf.validate_interval=1000 \
    ++train_conf.save_checkpoint_interval=1000 \
    ++train_conf.keep_nbest_models=10 \
    ++train_conf.avg_nbest_model=5 \
    ++train_conf.use_deepspeed=false \
    ++train_conf.use_bf16=true \
    ++enable_tf32=true \
    ++train_conf.deepspeed_config=${deepspeed_config} \
    ++optim_conf.lr=${LR} \
    ++output_dir="${OUTPUT_DIR}" &> ${log_file}

    if [ $? -eq 0 ]; then
        echo "✓ Stage ${STAGE_NAME} completed successfully! Model: ${OUTPUT_DIR}/model.pt.best"
    else
        echo "✗ Stage ${STAGE_NAME} failed. Check log: ${log_file}"
        exit 1
    fi
}

# ----------------------
# Stage 1: Warmup (50% general + 50% domain)
train_run "Stage1_Warmup" \
"${data_dir}/stage1/train.jsonl" \
"${data_dir}/stage1/val.jsonl" \
6 0.00003 \
"./outputs/stage1_warmup" \
"++model=${model_name_or_model_dir}"

# Stage 2: Domain Adaptation (20% general + 80% domain)
train_run "Stage2_Adaptation" \
"${data_dir}/stage2/train.jsonl" \
"${data_dir}/stage2/val.jsonl" \
6 0.00008 \
"./outputs/stage2_adaptation" \
"++model=${stage1_ckpt}" 

# Stage 3: Fine-tuning (100% domain) LoRA 微调
train_run "Stage3_Finetune" \
"${data_dir}/stage3/train.jsonl" \
"${data_dir}/stage3/val.jsonl" \
8 0.0002 \
"./outputs/stage3_finetune" \
"++model=${stage2_ckpt}" 

echo "All three stages completed successfully!"
