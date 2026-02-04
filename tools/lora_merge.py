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
STAGE3_CKPT = "outputs/stage3_finetune/model.pt.best"  # Stage 3 checkpoint
OUT_DIR = "models/Fun-ASR-Nano-merged"

print("=" * 70)
print("LoRA 合并脚本（修复版）")
print("=" * 70)

# ===== 1. 检查 checkpoint 是否包含 LoRA =====
print("\n[1] 检查 checkpoint...")
ckpt = torch.load(STAGE3_CKPT, map_location="cpu")

lora_keys = [k for k in ckpt.keys() if "lora" in k.lower()]
adaptor_keys = [k for k in ckpt.keys() if "adaptor" in k.lower()]

print(f"  Checkpoint 内容:")
print(f"    总参数: {len(ckpt)}")
print(f"    Adaptor: {len(adaptor_keys)}")
print(f"    LoRA: {len(lora_keys)}")

if len(lora_keys) == 0:
    print("\n⚠️  Checkpoint 不包含 LoRA 参数！")
    print("   原因: 训练时 LoRA 被 excludes 排除了")
    print("\n你有两个选择:")
    print("  1. 直接使用当前模型（adaptor 可能已经足够好）")
    print("  2. 重新训练 Stage 3 并修改保存策略")

    # 测试当前模型
    print("\n尝试加载模型测试...")
    try:
        test_model = AutoModel(
            model=BASE_MODEL_DIR,
            init_param=STAGE3_CKPT,
            trust_remote_code=True,
            device="cpu",
        )
        print("✓ 模型加载成功（仅 adaptor）")
        print("  你可以直接使用这个模型进行推理")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")

    exit(0)

# ===== 2. 如果有 LoRA，加载模型并应用 LoRA =====
print("\n[2] 加载基础模型并应用 LoRA...")

# 必须在加载时指定 LoRA 配置
model = AutoModel(
    model=BASE_MODEL_DIR,
    trust_remote_code=True,
    device="cpu",
    llm_conf=dict(
        use_lora=True,
        lora_conf=dict(
            r=32,
            lora_alpha=64,
        ),
    ),
)

net = model.model

# ===== 3. 加载训练好的权重 =====
print("\n[3] 加载训练权重...")
missing, unexpected = net.load_state_dict(ckpt, strict=False)

print(f"  Missing: {len(missing)}")
print(f"  Unexpected: {len(unexpected)}")

# 验证 LoRA 是否存在
lora_modules = []
for name, module in net.named_modules():
    if hasattr(module, "merge_and_unload"):
        lora_modules.append(name)

print(f"\n  检测到 {len(lora_modules)} 个 LoRA 模块")

if len(lora_modules) == 0:
    print("✗ 没有 LoRA 模块可以合并！")
    exit(1)

# ===== 4. 合并 LoRA =====
print("\n[4] 合并 LoRA 到基础权重...")
merged = 0
for name, module in net.named_modules():
    if hasattr(module, "merge_and_unload"):
        print(f"  合并: {name}")
        module.merge_and_unload()
        merged += 1

print(f"\n✓ 成功合并 {merged} 个 LoRA 模块")

# ===== 5. 保存合并后的模型 =====
print("\n[5] 保存合并后的模型...")
os.makedirs(OUT_DIR, exist_ok=True)

# 保存完整的 state_dict
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

print(f"\n✓ 合并完成！保存到: {OUT_DIR}")
print(f"  文件大小: {os.path.getsize(f'{OUT_DIR}/model.pt') / (1024*1024):.2f} MB")

# ===== 6. 验证合并后的模型 =====
print("\n[6] 验证合并后的模型...")
try:
    merged_model = AutoModel(model=OUT_DIR, trust_remote_code=True, device="cpu")
    print("✓ 合并后的模型可以正常加载")
except Exception as e:
    print(f"✗ 合并后的模型加载失败: {e}")

print("\n" + "=" * 70)
