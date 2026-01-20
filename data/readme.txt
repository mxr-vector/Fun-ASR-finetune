训练:验证:测试 = 8:1:1
G1-G66590 train集
G66591-G74915 valid集
G74916-G83238 test集
共计83238条，此外为了增强模型泛化能力 
专业数据与通用数据比例为8:2
因此额外增添20810条通用数据

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