#!/usr/bin/env python3
"""
正确的推理脚本 - 加载完整的三阶段训练结果
"""
from funasr import AutoModel
import sys


def load_finetuned_model(
    base_model_dir="models/Fun-ASR-Nano-2512",
    stage3_adaptor="models/lora_ckpt/model.pt.best",
    use_lora=True,
    device="cuda",
):
    """
    加载完整的微调模型

    参数说明:
    - base_model_dir: 原始模型（包含 Qwen3-0.6B）
    - stage3_adaptor: Stage 3 训练的 adaptor 权重
    - use_lora: 是否启用 LoRA（必须与训练时一致）
    """

    import torch
    
    # 1. 即使是推理，也必须先加载完整的 Base Model
    # 否则 AutoModel 会只加载 init_param 而忽略 Base Model 的 LLM 权重
    print(f"Loading base model from: {base_model_dir}")
    model = AutoModel(
        model=base_model_dir,
        trust_remote_code=True,
        device=device,
        # 必须指定 LLM 配置，确保结构正确
        llm_conf=dict(
            use_lora=use_lora,
            lora_conf=dict(
                r=32,
                lora_alpha=64,
            ) if use_lora else {},
        ),
    )

    # 2. 手动加载训练好的权重 (Adaptor + LoRA)
    if stage3_adaptor:
        print(f"Loading finetuned weights from: {stage3_adaptor}")
        state_dict = torch.load(stage3_adaptor, map_location="cpu")
        
        # 兼容完整 checkpoint
        if isinstance(state_dict, dict):
             if "model" in state_dict:
                 state_dict = state_dict["model"]
             elif "state_dict" in state_dict:
                 state_dict = state_dict["state_dict"]

        # 加载权重 (strict=False 是必须的，因为 Base Model 有很多非训练参数)
        model.model.load_state_dict(state_dict, strict=False)
        print("✓ Finetuned weights loaded successfully")

    return model


if __name__ == "__main__":
    # 加载模型
    print("加载微调后的模型...")
    model = load_finetuned_model()

    # 测试音频
    # wav_path = "data/test/gz1.wav"
    wav_path = f"{model.model_path}/example/zh.mp3"
    print(f"\n识别音频: {wav_path}")
    result = model.generate(input=wav_path)

    # 输出结果
    if isinstance(result, list):
        for r in result:
            print(f"结果: {r.get('text', '')}")
    else:
        print(f"结果: {result.get('text', '')}")
