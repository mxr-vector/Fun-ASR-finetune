#!/usr/bin/env python3
"""
正确的推理脚本 - 针对 Qwen3-ASR 的转写评估 demo
支持加载合并后的模型，或者原始模型 + LoRA Adapter
"""
import os
import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel


def load_finetuned_model(
    base_model_dir="models/qwen3-asr-merged",
    lora_ckpt=None,  # 如果使用未合并的 lora，可以指定这里，例如 "models/lora_ckpt/qwen3-asr-adapter-01"
    device="cuda",
):
    """
    加载完整的 Qwen3-ASR 模型
    """
    print(f"Loading base model from: {base_model_dir}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # 根据硬件选择 device_map 或手动 to(device)
    device_map = device if str(device).startswith("cuda") else "auto"

    model = Qwen3ASRModel.from_pretrained(
        base_model_dir,
        dtype=dtype,
        device_map=device_map,
    )
    
    if lora_ckpt and os.path.exists(lora_ckpt):
        print(f"Loading LoRA adapter from: {lora_ckpt}")
        model.model.thinker = PeftModel.from_pretrained(
            model.model.thinker,
            lora_ckpt,
            is_trainable=False,
        )
        print("✓ LoRA adapter loaded successfully")
        
    model.model.eval()

    # 消除 open-end generation 中 pad_token_id 的警告提示
    try:
        generation_config = getattr(model.model, "generation_config", None)
        if generation_config and getattr(generation_config, "pad_token_id", None) is None:
            generation_config.pad_token_id = getattr(generation_config, "eos_token_id", 151645)

        thinker = getattr(model.model, "thinker", None)
        if thinker and hasattr(thinker, "generation_config"):
            if getattr(thinker.generation_config, "pad_token_id", None) is None:
                thinker.generation_config.pad_token_id = getattr(thinker.generation_config, "eos_token_id", 151645)
    except Exception:
        pass

    return model


if __name__ == "__main__":
    print("加载 Qwen3-ASR 模型...")
    # ======================================================================
    # 默认假设你使用了 lora_merge_qwen3asr.py 合并了模型，并保存在 models/qwen3-asr-merged
    # 如果没有合并，你可以修改参数，例如：
    # model = load_finetuned_model(
    #     base_model_dir="models/Qwen3-ASR-1.7B", 
    #     lora_ckpt="models/lora_ckpt/qwen3-asr-adapter-01"
    # )
    # ======================================================================
    model = load_finetuned_model(
        base_model_dir="models/qwen3-asr-merged",
        lora_ckpt=None
    )

    test_dir = "data/test"

    # ====== 统一结果收集 ======
    all_results = []

    if not os.path.exists(test_dir):
        print(f"测试目录 {test_dir} 不存在，已自动尝试创建，请准备音频文件。")
        os.makedirs(test_dir, exist_ok=True)
    else:
        for file_name in os.listdir(test_dir):
            if file_name.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
                wav_path = os.path.join(test_dir, file_name)
                print(f"Processing: {wav_path}")

                try:
                    # Qwen3-ASR 使用 transcribe 进行推理
                    res = model.transcribe(audio=wav_path)
                except Exception as e:
                    print(f"Failed to process {wav_path}: {e}")
                    all_results.append(
                        {"file": file_name, "text": "", "status": "failed", "error": str(e)}
                    )
                    continue

                # 统一结构化存储（Qwen3-ASR 的返回结果每个对象通常具有 text 属性）
                if isinstance(res, list):
                    for r in res:
                        # 健壮地获取文本
                        text = getattr(r, "text", "") if hasattr(r, "text") else str(r)
                        all_results.append(
                            {"file": file_name, "text": text, "status": "ok"}
                        )
                else:
                    text = getattr(res, "text", "") if hasattr(res, "text") else str(res)
                    all_results.append(
                        {"file": file_name, "text": text, "status": "ok"}
                    )

        # ====== 最后统一输出 ======
        print("\n================ 统一识别结果 ================\n")
        if not all_results:
            print(f"未在 {test_dir} 中找到支持的音频文件。")
        else:
            for item in all_results:
                print(f"[{item['file']}] -> {item['text']}")

            # 可选：同时保存为文本文件
            output_file = "qwen3_asr_result.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                for item in all_results:
                    f.write(f"{item['file']}\t{item['text']}\n")

            print(f"\n✓ 全部处理完成，结果已写入 {output_file}")
