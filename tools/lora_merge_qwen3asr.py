#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 合并脚本 — 将 LoRA adapter 合并回 Qwen3-ASR 原始模型
"""
import argparse
import os

import torch

# ===== 配置 =====
BASE_MODEL_DIR = "models/Qwen3-ASR-1.7B"
STAGE3_CKPT = (
    "models/lora_ckpt/qwen3-asr-adapter-01"  # Stage 3 checkpoint
)
OUT_DIR = "models/qwen3-asr-merged"


def main():
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel

    # 1. 加载原始模型
    print(f"[1/4] 加载原始模型: {BASE_MODEL_DIR}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        BASE_MODEL_DIR,
        dtype=dtype,
        device_map="cpu",
    )
    model = asr_wrapper.model

    # 2. 加载 LoRA adapter 到 thinker
    print(f"[2/4] 加载 LoRA adapter: {STAGE3_CKPT}")
    model.thinker = PeftModel.from_pretrained(
        model.thinker,
        STAGE3_CKPT,
        is_trainable=False,
    )

    # 3. 合并权重
    print("[3/4] 合并 LoRA 权重...")
    model.thinker = model.thinker.merge_and_unload()

    # 4. 保存完整模型
    print(f"[4/4] 保存合并后的模型到: {OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR, safe_serialization=True)

    # 复制 processor 文件
    import shutil

    processor_files = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in processor_files:
        src = os.path.join(BASE_MODEL_DIR, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT_DIR, fn))

    print("=" * 50)
    print("✓ 合并完成！合并后的模型位于:")
    print(f"  {OUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
