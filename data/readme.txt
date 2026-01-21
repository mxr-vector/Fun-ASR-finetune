训练:验证:测试 = 8:1:1
G1-G66590 train集
G66591-G74915 valid集
G74916-G83238 test集
共计83238条，总时长约87h。此外为了增强模型泛化能力 
调取符合业务场景的WenetSpeech数据集，且数据集较小，仅调试音频适配器层
1.预热训练
通用数据：专业数据 = 50:50  87h:87h
训练轮数：5-10 epochs
目标：激活模型对多样化语音的适应能力

2.领域适配
通用数据：专业数据 = 20:80  20h:87h
训练轮数：15-20 epochs
目标：在保持泛化的前提下强化专业特征

3.纯专业数据：100%  87h
训练轮数：5-10 epochs
目标：最大化领域准确率


# linux
uv run tools/scp2jsonl.py \
  ++scp_file=data/zh/train/wav.scp \
  ++transcript_file=data/zh/train/wav.txt \
  ++jsonl_file=data/zh/train/wav.jsonl
# win train
uv run tools/scp2jsonl.py ++scp_file=data/zh/train/wav.scp ++transcript_file=data/zh/train/wav.txt ++jsonl_file=data/zh/train/wav.jsonl
# win valid
uv run tools/scp2jsonl.py ++scp_file=data/zh/valid/wav.scp ++transcript_file=data/zh/valid/wav.txt ++jsonl_file=data/zh/valid/wav.jsonl
# win test
uv run tools/scp2jsonl.py ++scp_file=data/zh/test/wav.scp ++transcript_file=data/zh/test/wav.txt ++jsonl_file=data/zh/test/wav.jsonl