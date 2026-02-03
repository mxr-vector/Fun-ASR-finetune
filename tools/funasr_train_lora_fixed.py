#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
修复版训练脚本 - 正确应用 LoRA
"""

import os
import sys
import torch
import hydra
import logging

from omegaconf import DictConfig
from torch.cuda.amp import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
import torch.distributed as dist

from funasr.register import tables
from funasr.train_utils.trainer_ds import Trainer
from funasr.train_utils.set_all_random_seed import set_all_random_seed
from funasr.utils.misc import prepare_model_dir
from funasr.train_utils.model_summary import model_summary
from funasr.train_utils.average_nbest_models import average_checkpoints
from funasr import AutoModel

# ===== 关键：导入 PEFT =====
try:
    from peft import get_peft_model, LoraConfig, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("peft not installed, LoRA will not be available")


def apply_lora_to_llm(model, lora_conf):
    """手动应用 LoRA 到 LLM"""
    if not PEFT_AVAILABLE:
        raise ImportError("peft is required for LoRA training")

    if not hasattr(model, "llm"):
        logging.warning("Model does not have 'llm' attribute, skipping LoRA")
        return model

    # ===== 应用到 model.llm（Qwen3ForCausalLM） =====
    llm_model = model.llm

    logging.info(f"LLM type: {type(llm_model)}")
    logging.info(f"LLM class: {llm_model.__class__.__name__}")

    # 配置 LoRA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_conf.get("r", 16),
        lora_alpha=lora_conf.get("lora_alpha", 32),
        lora_dropout=lora_conf.get("lora_dropout", 0.05),
        target_modules=lora_conf.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_conf.get("bias", "none"),
    )

    logging.info(f"Applying LoRA with config: {config}")

    # 应用 LoRA
    try:
        llm_with_lora = get_peft_model(llm_model, config)
        model.llm = llm_with_lora

        # 打印可训练参数
        llm_with_lora.print_trainable_parameters()

        logging.info("✓ LoRA applied successfully")
    except Exception as e:
        logging.error(f"✗ Failed to apply LoRA: {e}")
        import traceback

        traceback.print_exc()
        raise

    return model


@hydra.main(config_name=None, version_base=None)
def main_hydra(kwargs: DictConfig):
    from funasr.download.download_model_from_hub import download_model

    assert "model" in kwargs
    if "model_conf" not in kwargs:
        logging.info(
            "download models from model hub: {}".format(kwargs.get("hub", "ms"))
        )
        kwargs = download_model(is_training=kwargs.get("is_training", True), **kwargs)

    main(**kwargs)


def main(**kwargs):
    # 设置随机种子
    set_all_random_seed(kwargs.get("seed", 0))
    torch.backends.cudnn.enabled = kwargs.get(
        "cudnn_enabled", torch.backends.cudnn.enabled
    )
    torch.backends.cudnn.benchmark = kwargs.get(
        "cudnn_benchmark", torch.backends.cudnn.benchmark
    )
    torch.backends.cudnn.deterministic = kwargs.get("cudnn_deterministic", True)
    torch.backends.cuda.matmul.allow_tf32 = kwargs.get("enable_tf32", True)

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if local_rank == 0:
        tables.print()

    use_ddp = world_size > 1
    use_fsdp = kwargs.get("use_fsdp", False)
    use_deepspeed = kwargs.get("use_deepspeed", False)

    if use_deepspeed:
        import deepspeed

        deepspeed.init_distributed(dist_backend=kwargs.get("backend", "nccl"))
    elif use_ddp or use_fsdp:
        dist.init_process_group(
            backend=kwargs.get("backend", "nccl"), init_method="env://"
        )
        torch.cuda.set_device(local_rank)

    # 加载模型
    logging.info("Build model, frontend, tokenizer")
    device = kwargs.get("device", "cuda")
    kwargs["device"] = "cpu"
    model = AutoModel(**kwargs)

    if rank == 0:
        prepare_model_dir(**kwargs)

    kwargs = model.kwargs
    kwargs["device"] = device
    tokenizer = kwargs["tokenizer"]
    frontend = kwargs["frontend"]
    model = model.model
    del kwargs["model"]

    # ===== 关键修复：应用 LoRA =====
    llm_conf = kwargs.get("llm_conf", {})
    use_lora = llm_conf.get("use_lora", False)

    if use_lora:
        if local_rank == 0:
            logging.info("=" * 70)
            logging.info("Applying LoRA to LLM")
            logging.info("=" * 70)

        lora_conf = llm_conf.get("lora_conf", {})
        model = apply_lora_to_llm(model, lora_conf)

        if local_rank == 0:
            logging.info("=" * 70)

    # freeze_param
    freeze_param = kwargs.get("freeze_param", None)
    if freeze_param is not None:
        if "," in freeze_param:
            freeze_param = eval(freeze_param)
        if not isinstance(freeze_param, (list, tuple)):
            freeze_param = (freeze_param,)
        logging.info("freeze_param is not None: %s", freeze_param)
        for t in freeze_param:
            for k, p in model.named_parameters():
                if k.startswith(t + ".") or k == t:
                    logging.info(f"Setting {k}.requires_grad = False")
                    p.requires_grad = False

    # ===== 添加：打印参数统计 =====
    if local_rank == 0:
        logging.info(f"{model_summary(model)}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lora_params = sum(
            p.numel() for n, p in model.named_parameters() if "lora" in n.lower()
        )

        logging.info("=" * 70)
        logging.info("Parameter Statistics:")
        logging.info(f"  Total:     {total_params:,}")
        logging.info(
            f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)"
        )
        logging.info(
            f"  LoRA:      {lora_params:,} ({100*lora_params/total_params:.2f}%)"
        )
        logging.info("=" * 70)

    # 创建 Trainer
    trainer = Trainer(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        use_ddp=use_ddp,
        use_fsdp=use_fsdp,
        device=kwargs["device"],
        excludes=kwargs.get("excludes", None),
        output_dir=kwargs.get("output_dir", "./exp"),
        **kwargs.get("train_conf"),
    )

    model = trainer.warp_model(model, **kwargs)
    kwargs["device"] = int(os.environ.get("LOCAL_RANK", 0))
    trainer.device = int(os.environ.get("LOCAL_RANK", 0))
    model, optim, scheduler = trainer.warp_optim_scheduler(model, **kwargs)

    # 数据加载
    logging.info("Build dataloader")
    dataloader_class = tables.dataloader_classes.get(
        kwargs["dataset_conf"].get("dataloader", "DataloaderMapStyle")
    )
    dataloader = dataloader_class(**kwargs)

    scaler = GradScaler(enabled=True) if trainer.use_fp16 else None
    scaler = ShardedGradScaler(enabled=trainer.use_fp16) if trainer.use_fsdp else scaler

    trainer.resume_checkpoint(
        model=model, optim=optim, scheduler=scheduler, scaler=scaler
    )

    early_stopping_patience = kwargs.get("train_conf", {}).get(
        "early_stopping_patience", 0
    )
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # 训练循环
    import time

    dataloader_tr, dataloader_val = None, None
    for epoch in range(trainer.start_epoch, trainer.max_epoch):
        time1 = time.perf_counter()

        for data_split_i in range(
            trainer.start_data_split_i, dataloader.data_split_num
        ):
            time_slice_i = time.perf_counter()
            dataloader_tr, dataloader_val = dataloader.build_iter(
                epoch, data_split_i=data_split_i, start_step=trainer.start_step
            )

            trainer.train_epoch(
                model=model,
                optim=optim,
                scheduler=scheduler,
                scaler=scaler,
                dataloader_train=dataloader_tr,
                dataloader_val=dataloader_val,
                epoch=epoch,
                data_split_i=data_split_i,
                data_split_num=dataloader.data_split_num,
                start_step=trainer.start_step,
            )
            trainer.start_step = 0

            device = next(model.parameters()).device
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

            time_escaped = (time.perf_counter() - time_slice_i) / 3600.0
            logging.info(
                f"rank: {local_rank}, time_escaped: {time_escaped:.3f} hours"
            )

        trainer.start_data_split_i = 0
        trainer.validate_epoch(
            model=model, dataloader_val=dataloader_val, epoch=epoch + 1
        )
        current_val = trainer.val_loss_avg

        if current_val < best_val_loss:
            best_val_loss = current_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logging.info(
                f"No improvement for {epochs_no_improve}/{early_stopping_patience} epochs"
            )

        if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break

        trainer.step_in_epoch = 0
        trainer.save_checkpoint(
            epoch + 1, model=model, optim=optim, scheduler=scheduler, scaler=scaler
        )

        time_escaped = (time.perf_counter() - time1) / 3600.0
        logging.info(f"Epoch {epoch+1} completed in {time_escaped:.3f} hours")
        trainer.train_acc_avg = 0.0
        trainer.train_loss_avg = 0.0

    if trainer.rank == 0:
        average_checkpoints(
            trainer.output_dir,
            trainer.avg_nbest_model,
            use_deepspeed=trainer.use_deepspeed,
        )

    trainer.close()


if __name__ == "__main__":
    main_hydra()
