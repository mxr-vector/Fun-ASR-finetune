import os

import hydra
import torch
import json
from omegaconf import DictConfig, ListConfig, OmegaConf


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    def to_plain_list(cfg_item):
        if isinstance(cfg_item, ListConfig):
            return OmegaConf.to_container(cfg_item, resolve=True)
        elif isinstance(cfg_item, DictConfig):
            return {k: to_plain_list(v) for k, v in cfg_item.items()}
        else:
            return cfg_item
    kwargs = to_plain_list(cfg)

    model_dir = kwargs.get("model_dir", "FunAudioLLM/Fun-ASR-Nano-2512")
    scp_file = kwargs["scp_file"]
    output_file = kwargs["output_file"]

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    from funasr import AutoModel

    try:
        model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            remote_code="./model.py",
            device=device,
        )
    except AssertionError as e:
        cfg_path = os.path.join(model_dir, "config.json") if isinstance(model_dir, str) else ""
        if cfg_path and os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_json = json.load(f)
                model_type = str(cfg_json.get("model_type", "")).lower()
                archs = cfg_json.get("architectures", []) or []
                if model_type == "qwen3_asr" or "Qwen3ASRForConditionalGeneration" in archs:
                    raise SystemExit(
                        f"{e}\n"
                        f"检测到你传入的是 Qwen3-ASR 模型目录：{model_dir}\n"
                        f"该模型不属于 funasr 的 AutoModel 注册表，请改用：\n"
                        f"  uv run decode_qwen3asr.py ++model_dir={model_dir} ++scp_file=... ++output_file=...\n"
                    )
            except Exception:
                pass
        raise

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(scp_file, "r", encoding="utf-8") as f1:
        with open(output_file, "w", encoding="utf-8") as f2:
            for line in f1:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    text = model.generate(input=[parts[1]], cache={}, batch_size=1)[0]["text"]
                    f2.write(f"{parts[0]}\t{text}\n")


if __name__ == "__main__":
    main_hydra()
