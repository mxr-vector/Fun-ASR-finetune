docker run --shm-size=8g --gpus=all -p 10098:10095 -it --privileged=true \
  -v $PWD/models:/workspace/models \
  -v $PWD/data:/workspace/FunASR/data \
  -v $PWD/outputs:/workspace/FunASR/examples/industrial_data_pretraining/paraformer/outputs \
  --name funasr-paraformer \
  funasr-paraformer:0.2.1

# 专业数据集
scp2jsonl \
++scp_file_list='["/workspace/FunASR/data/domain/train/wav.scp", "/workspace/FunASR/data/domain/train/wav.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="/workspace/FunASR/data/domain/train/wav.jsonl"

scp2jsonl \
++scp_file_list='["/workspace/FunASR/data/domain/valid/wav.scp", "/workspace/FunASR/data/domain/valid/wav.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="/workspace/data/domain/valid/wav.jsonl"

#  通用数据集
scp2jsonl \
++scp_file_list='["/workspace/FunASR/data/general/train/wav.scp", "/workspace/FunASR/data/general/train/wav.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="/workspace/FunASR/data/general/train/wav.jsonl"

scp2jsonl \
++scp_file_list='["/workspace/FunASR/data/general/valid/wav.scp", "/workspace/FunASR/data/general/valid/wav.txt"]' \
++data_type_list='["source", "target"]' \
++jsonl_file_out="/workspace/FunASR/data/general/valid/wav.jsonl"


```bash
python prepare_staged_data.py \
  --general_train data/general/train/wav.jsonl \
  --general_val data/general/valid/wav.jsonl \
  --domain_train data/domain/train/wav.jsonl \
  --domain_val data/domain/valid/wav.jsonl \
  --output_dir data/staged
```

cd /workspace/FunASR/examples/industrial_data_pretraining/paraformer

nohup bash finetune.sh > full_train.log 2>&1 &