#!/usr/bin/env python3
import torch
from funasr import AutoModel

print("=" * 70)
print("诊断模型加载")
print("=" * 70)

# 1. 加载原始模型，看 LLM 如何初始化
print("\n[1] 加载原始模型...")
model_original = AutoModel(
    model="models/Fun-ASR-Nano-2512", trust_remote_code=True, device="cpu"
)

print("\n检查 LLM 参数来源:")
net = model_original.model

# 找一个 LLM 参数看看值
llm_params = [
    (name, param)
    for name, param in net.named_parameters()
    if "llm.model.layers.0.self_attn.q_proj.weight" in name
]

if llm_params:
    name, param = llm_params[0]
    print(f"  参数名: {name}")
    print(f"  形状: {param.shape}")
    print(f"  前几个值: {param.flatten()[:10]}")
    print(f"  均值: {param.mean().item():.6f}")
    print(f"  标准差: {param.std().item():.6f}")
else:
    print("  未找到 LLM 参数！")

# 2. 加载 Stage 2 模型
print("\n[2] 加载 Stage 2 checkpoint...")
stage2_ckpt = torch.load("outputs/stage1/model.pt.best", map_location="cpu")

llm_keys_in_stage2 = [k for k in stage2_ckpt.keys() if k.startswith("llm.")]
print(f"  Stage 2 中的 LLM 参数数量: {len(llm_keys_in_stage2)}")

if llm_keys_in_stage2:
    print("  示例 LLM 参数:")
    for k in llm_keys_in_stage2[:5]:
        print(f"    {k}: {stage2_ckpt[k].shape}")
else:
    print("  ⚠️  Stage 2 checkpoint 不包含 LLM 参数！")
    print("  这意味着 init_param 只会加载 encoder/adaptor")

# 3. 模拟 Stage 3 加载
print("\n[3] 模拟 Stage 3 加载过程...")
model_stage3 = AutoModel(
    model="models/Fun-ASR-Nano-2512",
    init_param="outputs/stage2/model.pt.best",
    trust_remote_code=True,
    device="cpu",
    llm_conf=dict(
        use_lora=True,
        lora_conf=dict(
            r=16,
            lora_alpha=32,
        ),
    ),
)

# 检查 LoRA 是否正确添加
net_stage3 = model_stage3.model
lora_params = [
    (name, param)
    for name, param in net_stage3.named_parameters()
    if "lora" in name.lower()
]

print(f"  LoRA 参数数量: {len(lora_params)}")
if lora_params:
    print("  LoRA 参数示例:")
    for name, param in lora_params[:5]:
        print(f"    {name}: {param.shape}, requires_grad={param.requires_grad}")
else:
    print("  ⚠️  没有找到 LoRA 参数！")

# 4. 检查 LLM 权重是否与原始模型一致
print("\n[4] 检查 LLM 权重来源...")
llm_param_stage3 = [
    (name, param)
    for name, param in net_stage3.named_parameters()
    if "llm.model.layers.0.self_attn.q_proj.weight" in name
]

if llm_param_stage3:
    name, param = llm_param_stage3[0]
    print(f"  Stage 3 LLM 参数统计:")
    print(f"    均值: {param.mean().item():.6f}")
    print(f"    标准差: {param.std().item():.6f}")

    # 与原始模型对比
    if llm_params:
        orig_param = llm_params[0][1]
        if torch.allclose(param, orig_param, rtol=1e-3):
            print("  ✓ LLM 权重来自原始 Qwen3-0.6B（正确）")
        else:
            print("  ✗ LLM 权重与原始模型不同（可能随机初始化）")

print("\n" + "=" * 70)
