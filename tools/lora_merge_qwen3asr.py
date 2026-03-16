#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import torch


BASE_MODEL_DIR = "models/Qwen3-ASR-1.7B"
STAGE3_CKPT = "models/lora_ckpt/qwen3-asr-adapter-01"
OUT_DIR = "models/qwen3-asr-merged"


def clean_generation_config(out_dir):
    """
    从磁盘层面彻底修复 generation_config.json
    """
    gen_path = os.path.join(out_dir, "generation_config.json")

    if not os.path.exists(gen_path):
        return

    with open(gen_path, "r", encoding="utf-8") as f:
        gen_cfg = json.load(f)

    # ===== 核心修复 =====
    if gen_cfg.get("do_sample", False) is False:
        gen_cfg.pop("temperature", None)
        gen_cfg.pop("top_p", None)
        gen_cfg.pop("top_k", None)

    # 可选：强制写死（更安全）
    gen_cfg["do_sample"] = False

    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, indent=2, ensure_ascii=False)

    print("✓ generation_config.json 已清理")


def main():
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel

    print(f"[1/4] 加载原始模型: {BASE_MODEL_DIR}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        BASE_MODEL_DIR,
        dtype=dtype,
        device_map="cpu",
    )
    model = asr_wrapper.model

    print(f"[2/4] 加载 LoRA adapter: {STAGE3_CKPT}")
    model.thinker = PeftModel.from_pretrained(
        model.thinker,
        STAGE3_CKPT,
        is_trainable=False,
    )

    print("[3/4] 合并 LoRA 权重...")
    model.thinker = model.thinker.merge_and_unload()

    print(f"[4/4] 保存合并后的模型到: {OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ===== 内存层修复（保留）=====
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        model.generation_config.do_sample = False

    model.save_pretrained(OUT_DIR, safe_serialization=True)

    # ===== 先复制 processor（注意顺序）=====
    import shutil

    processor_files = [
        "config.json",
        "generation_config.json",  # 会覆盖
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
        dst = os.path.join(OUT_DIR, fn)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # ===== 🔥 关键：最后再清理磁盘配置 =====
    clean_generation_config(OUT_DIR)

    print("=" * 50)
    print("✓ 合并完成！")
    print(f"  {OUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
