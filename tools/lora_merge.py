import torch
from funasr import AutoModel

# ===== 路径配置 =====
BASE_MODEL_DIR = "models/Fun-ASR-Nano-2512"  # 必须是官方/原始完整模型
LORA_CKPT = "models/lora_ckpt/model.pt.best" # 训练好的模型
OUT_DIR = "models/Fun-ASR-Nano-merged" # 输出目录

# ===== 1. 加载 base 模型（含完整 LLM）=====
model = AutoModel(model=BASE_MODEL_DIR, trust_remote_code=True, device="cpu")

net = model.model  # 真实的 torch.nn.Module

# ===== 2. 加载 LoRA 权重（非严格）=====
lora_state = torch.load(LORA_CKPT, map_location="cpu")
missing, unexpected = net.load_state_dict(lora_state, strict=False)

print("Missing keys (expected):", len(missing))
print("Unexpected keys:", unexpected)

# ===== 3. 合并 LoRA =====
merged = 0
for m in net.modules():
    if hasattr(m, "merge_and_unload"):
        m.merge_and_unload()
        merged += 1

print(f"Merged LoRA modules: {merged}")

# ===== 4. 保存完整模型 =====
import os

os.makedirs(OUT_DIR, exist_ok=True)
torch.save(net.state_dict(), f"{OUT_DIR}/model.pt")

print("Merged model saved to:", f"{OUT_DIR}/model.pt")
