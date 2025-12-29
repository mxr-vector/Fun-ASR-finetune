# Finetune

## Requirements

```
pip install git+https://github.com/modelscope/FunASR
```

## Data Prepare

Data examples

```
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "语音转写：<|startofspeech|>!https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav<|endofspeech|>"}, {"role": "assistant", "content": "甚至出现交易几乎停滞的情况"}], "speech_length": 418, "text_length": 6}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "语音转写：<|startofspeech|>!https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0916W0489.wav<|endofspeech|>"}, {"role": "assistant", "content": "湖北一公司以员工名义贷款数十员工负债千万"}], "speech_length": 572, "text_length": 11}
```

Full ref to `data/train_example.jsonl`

Description：

- `messages[1]["content"]`: audio file with speech recognition prompt
- `messages[2]["content"]`: transcription
- `speech_length`: number of fbank frames of the audio file
- `text_length`: number of tokens of the transcription (tokenized by `Qwen3-0.6B`)

`train_text.txt`

```
BAC009S0764W0121 甚至出现交易几乎停滞的情况
BAC009S0916W0489 湖北一公司以员工名义贷款数十员工负债千万
```

`train_wav.scp`

```
BAC009S0764W0121 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav
BAC009S0916W0489 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0916W0489.wav
```

`Command`

```
python tools/scp2jsonl.py \
  --scp-file /path/to/train_wav.scp \
  --transcript-file /path/to/train_text.txt \
  --jsonl-file data/train_example.jsonl
```

## Finetune

```
bash finetune.sh
```
