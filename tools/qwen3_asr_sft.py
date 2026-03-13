# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    required = [
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
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        return control


# ─── LoRA 相关 ────────────────────────────────────────────────────────────────

def apply_lora_to_model(model, args_cli):
    """对 thinker 的 LLM 部分应用 LoRA，冻结 audio_tower"""
    from peft import LoraConfig, TaskType, get_peft_model

    thinker = model.thinker

    # 1) 冻结 audio_tower 的所有参数
    for p in thinker.audio_tower.parameters():
        p.requires_grad = False
    print("[LoRA] ✓ audio_tower 已冻结")

    # 2) 冻结 lm_head
    if hasattr(thinker, "lm_head"):
        for p in thinker.lm_head.parameters():
            p.requires_grad = False
        print("[LoRA] ✓ lm_head 已冻结")

    # 3) 解析 target_modules
    target_modules = args_cli.lora_target_modules
    if isinstance(target_modules, str):
        target_modules = [m.strip() for m in target_modules.split(",")]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args_cli.lora_r,
        lora_alpha=args_cli.lora_alpha,
        lora_dropout=args_cli.lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=None,
    )

    # 4) 对 thinker 整体应用 LoRA（peft 会自动只在匹配 target_modules 的 Linear 上插入 adapter）
    thinker = get_peft_model(thinker, lora_config)
    model.thinker = thinker

    # 5) 关键：启用 input_require_grads，解决 LoRA + gradient checkpointing 兼容性
    #    LoRA 冻结了 embed_tokens，导致 embedding 输出无 requires_grad，
    #    gradient checkpointing 需要输入有 requires_grad 才能反向传播
    model.thinker.enable_input_require_grads()
    print("[LoRA] ✓ enable_input_require_grads 已启用")

    # 6) 确保 audio_tower 参数依然冻结（get_peft_model 不会影响它们，但双保险）
    for p in model.thinker.base_model.model.audio_tower.parameters():
        p.requires_grad = False

    # 7) 打印参数统计
    thinker.print_trainable_parameters()

    return model


class LoRACheckpointCallback(TrainerCallback):
    """LoRA 模式下，在每个 checkpoint 保存时额外保存 adapter 权重"""
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        # 保存 adapter 权重
        model = kwargs.get("model")
        if model is not None and hasattr(model, "thinker"):
            thinker = model.thinker
            # 如果是 DDP 包裹的，取 module
            if hasattr(thinker, "module"):
                thinker = thinker.module
            if hasattr(thinker, "save_pretrained"):
                adapter_dir = os.path.join(ckpt_dir, "adapter")
                thinker.save_pretrained(adapter_dir)
                print(f"[LoRA] ✓ Adapter 权重已保存到 {adapter_dir}")

        # 复制推理所需的配置文件
        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        return control


# ─── 参数解析 ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # Train hyper-params
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)

    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # Save
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    # LoRA
    p.add_argument("--use_lora", type=int, default=0, help="是否使用 LoRA 微调 (0/1)")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--lora_target_modules", type=str,
                   default="q_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help="LoRA target modules, 逗号分隔")
    p.add_argument("--lora_adapter_path", type=str, default="",
                   help="上一阶段的 LoRA adapter 路径，用于多阶段续训")

    # Gradient checkpointing
    p.add_argument("--gradient_checkpointing", type=int, default=0,
                   help="是否启用 gradient checkpointing (0/1)")

    return p.parse_args()


def main():
    args_cli = parse_args()

    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    # ── LoRA 应用 ──
    use_lora = args_cli.use_lora == 1
    if use_lora:
        print("=" * 70)
        print("[LoRA] 正在应用 LoRA 微调...")
        print(f"[LoRA] r={args_cli.lora_r}, alpha={args_cli.lora_alpha}, "
              f"dropout={args_cli.lora_dropout}")
        print(f"[LoRA] target_modules={args_cli.lora_target_modules}")
        print("=" * 70)

        adapter_path = (args_cli.lora_adapter_path or "").strip()
        if adapter_path:
            # 多阶段续训：加载前一阶段的 adapter 权重
            from peft import PeftModel
            print(f"[LoRA] 多阶段续训: 加载前一阶段 adapter 从 {adapter_path}")
            model.thinker = PeftModel.from_pretrained(
                model.thinker, adapter_path, is_trainable=True,
            )
            model.thinker.enable_input_require_grads()
            # 冻结 audio_tower（双保险）
            for p in model.thinker.base_model.model.audio_tower.parameters():
                p.requires_grad = False
            if hasattr(model.thinker.base_model.model, "lm_head"):
                for p in model.thinker.base_model.model.lm_head.parameters():
                    p.requires_grad = False
            model.thinker.print_trainable_parameters()
            print(f"[LoRA] ✓ 前一阶段 adapter 加载完成")
        else:
            # 首次训练：新建 LoRA adapter
            model = apply_lora_to_model(model, args_cli)

    # ── Gradient Checkpointing ──
    if args_cli.gradient_checkpointing == 1:
        if use_lora:
            # LoRA 模式：仅对 text model (decoder) 启用，跳过已冻结的 audio_tower
            # PeftModel 结构: model.thinker.base_model.model = 原始 thinker
            #   原始 thinker.model = Qwen3ASRThinkerTextModel (decoder)
            #   原始 thinker.audio_tower = 音频编码器 (已冻结，不应开启 grad ckpt)
            base_thinker = model.thinker.base_model.model  # 原始 thinker
            if hasattr(base_thinker, "model"):  # .model = TextModel (decoder)
                base_thinker.model.gradient_checkpointing = True
                print("[GradCkpt] ✓ text model (decoder) gradient_checkpointing 已启用")
                print("[GradCkpt]   audio_tower 跳过（已冻结）")
        else:
            # 全量微调模式：对整个 thinker 启用
            if hasattr(model.thinker, "gradient_checkpointing_enable"):
                model.thinker.gradient_checkpointing_enable()
                print("[GradCkpt] ✓ thinker gradient_checkpointing 已启用")

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({
                "validation": args_cli.eval_file} if args_cli.eval_file else {}),
        },
    )
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForQwen3ASRFinetuning(processor=processor, sampling_rate=args_cli.sr)

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1),
        dataloader_prefetch_factor=args_cli.prefetch_factor if args_cli.num_workers > 0 else None,
        save_strategy=args_cli.save_strategy,
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args_cli.save_steps,
        do_eval=bool(args_cli.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=True if use_lora else False,
        remove_unused_columns=False,
        report_to="none",
        # LoRA 模式下不在 TrainingArguments 中开启 gradient_checkpointing，
        # 因为我们已手动仅对 text model 启用，避免 Trainer 对冻结的 audio_tower 也启用
        gradient_checkpointing=(args_cli.gradient_checkpointing == 1) and (not use_lora),
    )

    # ── 回调 ──
    callbacks = []
    if use_lora:
        callbacks.append(LoRACheckpointCallback(base_model_path=args_cli.model_path))
    else:
        callbacks.append(MakeEveryCheckpointInferableCallback(base_model_path=args_cli.model_path))

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=callbacks,
    )

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # ── LoRA 模式：训练结束后保存最终 adapter 至 best_model/adapter ──
    if use_lora and trainer.args.process_index == 0:
        best_model_dir = os.path.join(args_cli.output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        adapter_dir = os.path.join(best_model_dir, "adapter")
        thinker = model.thinker
        if hasattr(thinker, "module"):
            thinker = thinker.module
        thinker.save_pretrained(adapter_dir)
        copy_required_hf_files_for_qwen_asr(args_cli.model_path, best_model_dir)
        print(f"[LoRA] ✓ 最终 adapter 已保存到 {adapter_dir}")


if __name__ == "__main__":
    main()
