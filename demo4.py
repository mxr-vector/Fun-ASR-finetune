#!/usr/bin/env python3
"""
正确的推理脚本 - 加载完整的三阶段训练结果
"""
from funasr import AutoModel
import sys


def load_finetuned_model(
    base_model_dir="models/Fun-ASR-Nano-2512",
    stage3_adaptor="models/Fun-ASR-Nano-merged/model.pt.best",
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

    if use_lora:
        # Stage 3 使用了 LoRA
        model = AutoModel(
            model=base_model_dir,
            init_param=stage3_adaptor,  # 加载 adaptor
            trust_remote_code=True,
            device=device,
            # 必须指定 LoRA 配置
            llm_conf=dict(
                use_lora=True,
                lora_conf=dict(
                    r=32,
                    lora_alpha=64,
                ),
            ),
        )
    else:
        # 如果没有使用 LoRA（Stage 1/2）
        model = AutoModel(
            model=base_model_dir,
            init_param=stage3_adaptor,
            trust_remote_code=True,
            device=device,
        )

    return model


if __name__ == "__main__":
    # 加载模型
    print("加载微调后的模型...")
    model = load_finetuned_model()

    # 测试音频
    test_audio = "data/test/gz1.wav"

    print(f"\n识别音频: {test_audio}")
    result = model.generate(input=test_audio)

    # 输出结果
    if isinstance(result, list):
        for r in result:
            print(f"结果: {r.get('text', '')}")
    else:
        print(f"结果: {result.get('text', '')}")
