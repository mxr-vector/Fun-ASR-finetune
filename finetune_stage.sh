#!/bin/bash
# finetune_stage.sh - 适配预混合数据的版本

workspace=`pwd`

STAGE=${1:-1}

echo "========================================"
echo "Training Stage: ${STAGE}"
echo "========================================"

export CUDA_VISIBLE_DEVICES="2,3"
# gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_num=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
else
    gpu_num=$(nvidia-smi -L | wc -l)
fi
echo "Using gpu_num = $gpu_num"

# 预训练模型路径
model_name_or_model_dir="models/Fun-ASR-Nano-2512"

# ============ 修改：直接指向预混合的数据 ============
data_dir="${workspace}/data/staged"

# 写死的检查点路径
stage1_best_model="./outputs/stage1_warmup/model.pt.best"
stage2_best_model="./outputs/stage2_adaptation/model.pt.best"

# 根据阶段配置
case ${STAGE} in
    1)
        echo "Stage 1: Warmup (50% general + 50% domain)"
        train_data="${data_dir}/stage1/train.jsonl"
        val_data="${data_dir}/stage1/val.jsonl"
        max_epoch=3
        learning_rate=0.00001
        output_dir="./outputs/stage1_warmup"
        MODEL_INIT_PARAM="++model=${model_name_or_model_dir}"
        # Stage 1: 只训练adaptor
        FREEZE_PARAMS="
++audio_encoder_conf.freeze=true \
++audio_adaptor_conf.freeze=false \
++llm_conf.freeze=true \
++llm_conf.use_lora=false \
++llm_conf.lora_conf.freeze_lora=true
"
        ;;
        
    2)
        echo "Stage 2: Domain Adaptation (20% general + 80% domain)"
        
        if [ ! -f "${stage1_best_model}" ]; then
            echo "ERROR: Stage 1 model not found: ${stage1_best_model}"
            exit 1
        fi
        
        train_data="${data_dir}/stage2/train.jsonl"
        val_data="${data_dir}/stage2/val.jsonl"
        max_epoch=3
        learning_rate=0.00005
        output_dir="./outputs/stage2_adaptation"
        MODEL_INIT_PARAM="++init_param=${stage1_best_model}"
        # Stage 2: 只训练adaptor
        FREEZE_PARAMS="
++audio_encoder_conf.freeze=true \
++audio_adaptor_conf.freeze=false \
++llm_conf.freeze=true \
++llm_conf.use_lora=false \
++llm_conf.lora_conf.freeze_lora=true
"
        ;;
        
    3)
        echo "Stage 3: Fine-tuning (100% domain)"
        
        if [ ! -f "${stage2_best_model}" ]; then
            echo "ERROR: Stage 2 model not found: ${stage2_best_model}"
            exit 1
        fi
        
        train_data="${data_dir}/stage3/train.jsonl"
        val_data="${data_dir}/stage3/val.jsonl"
        max_epoch=4
        learning_rate=0.0002
        output_dir="./outputs/stage3_finetune"
        MODEL_INIT_PARAM="++init_param=${stage2_best_model}"
        # Stage 3: 冻结encoder. LoRA微调LLM
        FREEZE_PARAMS="
++audio_encoder_conf.freeze=true \
++audio_adaptor_conf.freeze=false \
++llm_conf.freeze=false \
++llm_conf.use_lora=true \
++llm_conf.lora_conf.freeze_lora=false
"
        ;;
        
    *)
        echo "ERROR: Invalid stage ${STAGE}"
        exit 1
        ;;
esac

log_file="${output_dir}/log.txt"

mkdir -p  ${output_dir}
echo "log_file: ${log_file}"
# -------------------------
# 判定同阶段续传， 不同阶段之间使用上一个阶段最好模型
# -------------------------
CURRENT_STAGE_CKPT="${output_dir}/model.pt"

if [ -f "${CURRENT_STAGE_CKPT}" ]; then
    echo ">> Found checkpoint: ${CURRENT_STAGE_CKPT}"
    echo ">> Resume training from this stage"
    INIT_PARAM="++model=${model_name_or_model_dir} ++init_param=${CURRENT_STAGE_CKPT}"
    RESUME_PARAM="++train_conf.resume=true"
else
    echo ">> No checkpoint found in ${output_dir}"
    echo ">> Init from previous stage model"
    INIT_PARAM="++model=${model_name_or_model_dir} ${MODEL_INIT_PARAM}"
    RESUME_PARAM="++train_conf.resume=false"
fi

DISTRIBUTED_ARGS="
    --nnodes ${WORLD_SIZE:-1} \
    --nproc_per_node $gpu_num \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-26669}
"

train_tool=`which funasr-train-ds`
deepspeed_config=${workspace}/deepspeed_conf/ds_stage1.json


echo "----------------------------------------"
echo "Train data : ${train_data}"
echo "Valid data : ${val_data}"
echo "Output dir : ${output_dir}"
echo "Resume     : ${RESUME_PARAM}"
echo "Init param : ${INIT_PARAM}"
echo "----------------------------------------"

# ============ 训练命令 ============
torchrun $DISTRIBUTED_ARGS \
${train_tool} \
${INIT_PARAM} \
${RESUME_PARAM} \
${FREEZE_PARAMS} \
++trust_remote_code=true \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset_conf.data_split_num=1 \
++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.batch_type="token" \
++dataset_conf.batch_size=12000 \
++dataset_conf.sort_size=1024 \
++dataset_conf.num_workers=4 \
++dataset_conf.shuffle=true \
++train_conf.max_epoch=${max_epoch} \
++train_conf.log_interval=1 \
++train_conf.validate_interval=1000 \
++train_conf.save_checkpoint_interval=1000 \
++train_conf.keep_nbest_models=10 \
++train_conf.avg_nbest_model=5 \
++train_conf.use_deepspeed=false \
++train_conf.use_bf16=true \
++enable_tf32=true \
++train_conf.deepspeed_config=${deepspeed_config} \
++optim_conf.lr=${learning_rate} \
++output_dir="${output_dir}" &> ${log_file}

# 训练完成
if [ $? -eq 0 ]; then
    echo "✓ Stage ${STAGE} completed successfully!"
    echo "  Model: ${output_dir}/model.pt.best"
    
    if [ ${STAGE} -eq 1 ]; then
        echo ""
        echo "Next: Stage 2"
    elif [ ${STAGE} -eq 2 ]; then
        echo ""
        echo "Next: Stage 3"
    else
        echo ""
        echo "All stages completed!"
    fi
    exit 0
else
    echo "✗ Stage ${STAGE} failed. Check: ${log_file}"
    exit 1
fi