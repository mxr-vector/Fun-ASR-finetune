#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 合并脚本 — 将 LoRA adapter 合并回 Qwen3-ASR 原始模型

用法:
    python tools/lora_merge_qwen3asr.py \
        --base_model_path   models/Qwen3-ASR-1.7B \
        --adapter_path      outputs/staged/stage3/best_model/adapter \
        --output_path       outputs/merged_model
"""
import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser("Merge LoRA adapter back into Qwen3-ASR base model")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="原始 Qwen3-ASR 模型路径")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="LoRA adapter 路径（包含 adapter_config.json 和 adapter_model.*）")
    parser.add_argument("--output_path", type=str, required=True,
                        help="合并后的完整模型输出路径")
    args = parser.parse_args()

    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel

    # 1. 加载原始模型
    print(f"[1/4] 加载原始模型: {args.base_model_path}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.base_model_path,
        dtype=dtype,
        device_map="cpu",
    )
    model = asr_wrapper.model

    # 2. 加载 LoRA adapter 到 thinker
    print(f"[2/4] 加载 LoRA adapter: {args.adapter_path}")
    model.thinker = PeftModel.from_pretrained(
        model.thinker,
        args.adapter_path,
        is_trainable=False,
    )

    # 3. 合并权重
    print("[3/4] 合并 LoRA 权重...")
    model.thinker = model.thinker.merge_and_unload()

    # 4. 保存完整模型
    print(f"[4/4] 保存合并后的模型到: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path, safe_serialization=True)

    # 复制 processor 文件
    import shutil
    processor_files = [
        "config.json", "generation_config.json",
        "preprocessor_config.json", "processor_config.json",
        "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "chat_template.json",
        "merges.txt", "vocab.json",
    ]
    for fn in processor_files:
        src = os.path.join(args.base_model_path, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_path, fn))

    print("=" * 50)
    print("✓ 合并完成！合并后的模型位于:")
    print(f"  {args.output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
