#!/usr/bin/env python3
"""
正确的推理脚本 - 加载完整的三阶段训练结果
model_with_adapter
"""
from funasr import AutoModel
import sys
import os
import torch


def load_finetuned_model(
    base_model_dir="models/Fun-ASR-Nano-2512",
    stage3_adaptor="models/lora_ckpt/model.pt.best",
    use_lora=True,
    device="cuda",
):
    """
    加载完整的微调模型
    """

    print(f"Loading base model from: {base_model_dir}")
    model = AutoModel(
        model=base_model_dir,
        trust_remote_code=True,
        device=device,
        llm_conf=dict(
            use_lora=use_lora,
            lora_conf=(
                dict(
                    r=32,
                    lora_alpha=64,
                )
                if use_lora
                else {}
            ),
        ),
    )

    if stage3_adaptor:
        print(f"Loading finetuned weights from: {stage3_adaptor}")
        state_dict = torch.load(stage3_adaptor, map_location="cpu")

        if isinstance(state_dict, dict):
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        model.model.load_state_dict(state_dict, strict=False)
        print("✓ Finetuned weights loaded successfully")

    return model


if __name__ == "__main__":
    print("加载微调后的模型...")
    model = load_finetuned_model()

    test_dir = "data/test"

    # ====== 统一结果收集 ======
    all_results = []

    for file_name in os.listdir(test_dir):
        if file_name.lower().endswith((".wav", ".WAV")):
            wav_path = os.path.join(test_dir, file_name)
            print(f"Processing: {wav_path}")

            try:
                res = model.generate(input=wav_path)
            except Exception as e:
                print(f"Failed to process {wav_path}: {e}")
                all_results.append(
                    {"file": file_name, "text": "", "status": "failed", "error": str(e)}
                )
                continue

            # 统一结构化存储
            if isinstance(res, list):
                for r in res:
                    all_results.append(
                        {"file": file_name, "text": r.get("text", ""), "status": "ok"}
                    )
            else:
                all_results.append(
                    {"file": file_name, "text": res.get("text", ""), "status": "ok"}
                )

    # ====== 最后统一输出 ======
    print("\n================ 统一识别结果 ================\n")
    for item in all_results:
        print(f"[{item['file']}] -> {item['text']}")

    # 可选：同时保存为文本文件
    with open("asr_result.txt", "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(f"{item['file']}\t{item['text']}\n")

    print("\n✓ 全部处理完成，结果已写入 asr_result.txt")
