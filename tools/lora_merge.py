#!/usr/bin/env python3
"""
合并 LoRA 到基础模型（修复版）
前提：checkpoint 必须包含 LoRA 参数
"""
import torch
import os
from funasr import AutoModel

# ===== 配置 =====
BASE_MODEL_DIR = "models/Fun-ASR-Nano-2512"
STAGE3_CKPT = "models/lora_ckpt/model.pt.best"  # Stage 3 checkpoint
OUT_DIR = "models/Fun-ASR-Nano-merged"

print("=" * 70)
print("LoRA 合并脚本（修复版）")
print("=" * 70)

# ===== 1. 检查 checkpoint =====
print("\n[1] 检查 checkpoint...")
full_ckpt = torch.load(STAGE3_CKPT, map_location="cpu")

# 提取真正的 state_dict
if "state_dict" in full_ckpt:
    ckpt = full_ckpt["state_dict"]
elif "model" in full_ckpt:
    ckpt = full_ckpt["model"]
else:
    ckpt = full_ckpt

lora_keys = [k for k in ckpt.keys() if "lora" in k.lower()]
adaptor_keys = [k for k in ckpt.keys() if "adaptor" in k.lower()]

print(f"  Checkpoint 内容:")
print(f"    原始文件 Key: {list(full_ckpt.keys())[:5]}...")
print(f"    提取后的参数总数: {len(ckpt)}")
print(f"    Adaptor 相关参数: {len(adaptor_keys)}")
print(f"    LoRA 相关参数: {len(lora_keys)}")

if len(lora_keys) == 0 and len(adaptor_keys) == 0:
    print("\n⚠️  Checkpoint 不包含已训练的参数 (LoRA 或 Adaptor)！")
    exit(1)

# ===== 2. 加载基础模型并手动应用 LoRA =====
print("\n[2] 加载基础模型并手动应用 LoRA...")

model = AutoModel(
    model=BASE_MODEL_DIR,
    trust_remote_code=True,
    device="cpu",
)

net = model.model

# 手动应用 LoRA 到 LLM
print("  正在应用 LoRA 结构到 LLM...")
try:
    from peft import get_peft_model, LoraConfig, TaskType
    
    # 这里的配置必须和训练时完全一致 (finetune_nano.sh Stage 3)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    
    net.llm = get_peft_model(net.llm, lora_config)
    print("  ✓ LoRA 结构已应用")
except Exception as e:
    print(f"  ✗ 应用 LoRA 失败: {e}")
    exit(1)

# ===== 3. 加载训练好的权重 =====
print("\n[3] 加载训练权重 (Base + Adaptor + LoRA)...")
missing, unexpected = net.load_state_dict(ckpt, strict=False)

print(f"  Missing keys: {len(missing)}")
print(f"  Unexpected keys: {len(unexpected)}")

# ===== 4. 合并 LoRA =====
print("\n[4] 合并 LoRA 到基础权重...")

if hasattr(net.llm, "merge_and_unload"):
    print("  正在执行 merge_and_unload()...")
    net.llm = net.llm.merge_and_unload()
    print("  ✓ LoRA 已合并到基础权重")
else:
    print("  ✗ 无法找到 merge_and_unload 方法！")
    exit(1)

# ===== 5. 保存合并后的模型 =====
print("\n[5] 保存合并后的模型...")
os.makedirs(OUT_DIR, exist_ok=True)

# 确保我们保存的是原始合并后的结构 (不含 PEFT 封装)
torch.save(net.state_dict(), f"{OUT_DIR}/model.pt")

# 复制配置文件
import shutil
for file in [
    "config.yaml",
    "configuration.json",
    "tokenizer_config.json",
    "multilingual.tiktoken",
]:
    src = os.path.join(BASE_MODEL_DIR, file)
    dst = os.path.join(OUT_DIR, file)
    if os.path.exists(src):
        # 如果是 config.yaml，我们要确保 use_lora 是 false (因为已经合并了)
        shutil.copy(src, dst)
        print(f"  复制: {file}")

# 复制 Qwen3-0.6B 目录
qwen_src = os.path.join(BASE_MODEL_DIR, "Qwen3-0.6B")
qwen_dst = os.path.join(OUT_DIR, "Qwen3-0.6B")
if os.path.exists(qwen_src):
    if os.path.exists(qwen_dst):
        shutil.rmtree(qwen_dst)
    shutil.copytree(qwen_src, qwen_dst)
    print(f"  复制: Qwen3-0.6B/")
# example
example_src = os.path.join(BASE_MODEL_DIR, "example")
example_dst = os.path.join(OUT_DIR, "example")
if os.path.exists(example_src):
    if os.path.exists(example_dst):
        shutil.rmtree(example_dst)
    shutil.copytree(example_src, example_dst)
    print(f"  复制: example/")

print(f"\n✓ 合并完成！保存到: {OUT_DIR}")
print(f"  最终模型参数量: {len(net.state_dict())}")

# ===== 6. 验证合并后的模型 =====
print("\n[6] 验证合并后的模型...")
try:
    # 验证时不需要 init_param，因为已经合并了
    merged_model = AutoModel(model=OUT_DIR, trust_remote_code=True, device="cpu", disable_update=True)
    print("✓ 合并后的模型可以正常加载")
except Exception as e:
    print(f"✗ 合并后的模型加载失败: {e}")

print("\n" + "=" * 70)
