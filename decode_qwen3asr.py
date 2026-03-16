import os
import sys
from typing import Any, Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


def _to_plain_obj(cfg_item: Any):
    if isinstance(cfg_item, ListConfig):
        return OmegaConf.to_container(cfg_item, resolve=True)
    if isinstance(cfg_item, DictConfig):
        return {k: _to_plain_obj(v) for k, v in cfg_item.items()}
    return cfg_item


def _pick_device(user_device: Optional[str] = None) -> str:
    if user_device:
        return str(user_device)

    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        raise ValueError("dtype is None")
    s = str(dtype).strip().lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if s not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}. Supported: {sorted(mapping)}")
    return mapping[s]


def _pick_dtype(device: str, user_dtype: Optional[Any] = None) -> torch.dtype:
    if user_dtype is not None and str(user_dtype).strip() != "":
        return _parse_dtype(user_dtype)

    if str(device).startswith("cuda"):
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    if str(device) == "mps":
        return torch.float16

    return torch.float32


def _first_int_token_id(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return int(x)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        for v in x:
            got = _first_int_token_id(v)
            if got is not None:
                return got
    return None


def _maybe_fix_pad_token_id(asr: Any, pad_token_id: Optional[int] = None) -> None:
    if getattr(asr, "backend", None) != "transformers":
        return

    model = getattr(asr, "model", None)
    if model is None:
        return

    processor = getattr(asr, "processor", None)
    tokenizer = getattr(processor, "tokenizer", None) if processor is not None else None

    desired_pad_id = _first_int_token_id(pad_token_id)
    if desired_pad_id is None:
        desired_pad_id = _first_int_token_id(
            getattr(getattr(model, "generation_config", None), "pad_token_id", None)
        )
    if desired_pad_id is None and tokenizer is not None:
        desired_pad_id = _first_int_token_id(getattr(tokenizer, "pad_token_id", None))
    if desired_pad_id is None and tokenizer is not None:
        desired_pad_id = _first_int_token_id(getattr(tokenizer, "eos_token_id", None))
    if desired_pad_id is None:
        desired_pad_id = _first_int_token_id(
            getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        )
    if desired_pad_id is None:
        try:
            import inspect

            sig = inspect.signature(model.generate)
            desired_pad_id = _first_int_token_id(
                sig.parameters.get("eos_token_id").default
            )
        except Exception:
            desired_pad_id = None

    if desired_pad_id is None:
        return

    # Fix outer model generation_config (not always used in this model's custom generate()).
    try:
        if (
            hasattr(model, "generation_config")
            and getattr(model.generation_config, "pad_token_id", None) is None
        ):
            model.generation_config.pad_token_id = desired_pad_id
    except Exception:
        pass

    # Qwen3-ASR overrides outer generate() and calls thinker.generate(), so fix thinker too.
    thinker = getattr(model, "thinker", None)
    if thinker is None:
        return

    try:
        if (
            not hasattr(thinker, "generation_config")
            or thinker.generation_config is None
        ):
            from transformers import GenerationConfig

            thinker.generation_config = GenerationConfig.from_model_config(
                thinker.config
            )
        if getattr(thinker.generation_config, "pad_token_id", None) is None:
            thinker.generation_config.pad_token_id = desired_pad_id
    except Exception:
        pass

    try:
        if (
            hasattr(thinker, "config")
            and getattr(thinker.config, "pad_token_id", None) is None
        ):
            thinker.config.pad_token_id = desired_pad_id
    except Exception:
        pass

    try:
        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = desired_pad_id
    except Exception:
        pass


def _transcribe_safe(
    asr: Any,
    audio_paths: List[str],
    *,
    context: str,
    language: Optional[str],
    write_language: bool,
) -> List[str]:
    def _format_one(r) -> str:
        if r is None:
            return ""
        if write_language:
            return f"{r.language}\t{r.text}"
        return r.text

    try:
        rs = asr.transcribe(audio=audio_paths, context=context, language=language)
        return [_format_one(r) for r in rs]
    except Exception as e:
        print(f"[WARN] batch transcribe failed: {e}", file=sys.stderr)
        outs: List[str] = []
        for p in audio_paths:
            try:
                r = asr.transcribe(audio=p, context=context, language=language)[0]
                outs.append(_format_one(r))
            except Exception as e2:
                print(f"[WARN] transcribe failed for {p}: {e2}", file=sys.stderr)
                outs.append(_format_one(None))
        return outs


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    kwargs: Dict[str, Any] = _to_plain_obj(cfg) if cfg is not None else {}

    model_dir = kwargs.get("model_dir", "models/Qwen3-ASR-1.7B")
    scp_file = kwargs["scp_file"]
    output_file = kwargs["output_file"]

    context = kwargs.get("context", "") or ""
    language = kwargs.get("language", None)
    if language is not None and str(language).strip() == "":
        language = None

    batch_size = int(kwargs.get("batch_size", 1))
    max_inference_batch_size = int(kwargs.get("max_inference_batch_size", 32))
    max_new_tokens = int(kwargs.get("max_new_tokens", 512))

    write_language = bool(kwargs.get("write_language", False))
    pad_token_id = kwargs.get("pad_token_id", None)

    device = _pick_device(kwargs.get("device", None))
    dtype = _pick_dtype(device=device, user_dtype=kwargs.get("dtype", None))

    from qwen_asr import Qwen3ASRModel

    asr = Qwen3ASRModel.from_pretrained(
        model_dir,
        dtype=dtype,
        device_map=None,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
    )
    if getattr(asr, "backend", None) == "transformers":
        asr.model.eval()
        asr.model.to(device)
        _maybe_fix_pad_token_id(asr, pad_token_id=_first_int_token_id(pad_token_id))

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # --- 新增：读取所有行以计算总进度 ---
    all_lines = []
    with open(scp_file, "r", encoding="utf-8") as f1:
        all_lines = [line.strip() for line in f1 if line.strip()]

    total_files = len(all_lines)
    print(f"总共需要处理 {total_files} 个音频文件。", file=sys.stdout)
    processed_count = 0
    # -------------------------------------

    keys: List[str] = []
    paths: List[str] = []
    with open(output_file, "w", encoding="utf-8") as f2:
        for line in all_lines:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue

            keys.append(parts[0])
            paths.append(parts[1])

            if len(keys) < batch_size:
                continue

            # 处理批次
            texts = _transcribe_safe(
                asr,
                paths,
                context=context,
                language=language,
                write_language=write_language,
            )
            for k, t in zip(keys, texts):
                f2.write(f"{k}\t{t}\n")

            f2.flush()

            # 更新并打印进度
            processed_count += len(keys)
            progress_percent = (processed_count / total_files) * 100
            print(
                f"进度: {processed_count} / {total_files} ({progress_percent:.2f}%)",
                end="\r",
                file=sys.stdout,
            )

            keys.clear()
            paths.clear()

        # 处理最后一个可能不满的批次
        if keys:
            texts = _transcribe_safe(
                asr,
                paths,
                context=context,
                language=language,
                write_language=write_language,
            )
            for k, t in zip(keys, texts):
                f2.write(f"{k}\t{t}\n")

            # 最终进度更新
            processed_count += len(keys)
            progress_percent = (processed_count / total_files) * 100
            print(
                f"进度: {processed_count} / {total_files} ({progress_percent:.2f}%)",
                file=sys.stdout,
            )

    # 打印完成信息
    print("\n转录任务已完成。", file=sys.stdout)


if __name__ == "__main__":
    main_hydra()
