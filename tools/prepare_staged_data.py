#!/usr/bin/env python3
"""
为三阶段训练准备混合数据
生成的是索引文件，不复制音频，节省空间
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import argparse


def load_jsonl(path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    """保存JSONL文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved: {path} ({len(data)} samples)")


def mix_datasets(
    general_data: List[Dict], domain_data: List[Dict], general_ratio: float
) -> List[Dict]:
    """
    混合数据集

    Args:
        general_data: 通用数据列表
        domain_data: 专业数据列表
        general_ratio: 通用数据占比 (0.0-1.0)

    Returns:
        混合后的数据列表
    """
    if general_ratio == 0:
        # 纯专业数据
        return domain_data.copy()

    # 计算需要采样的通用数据量
    # 例如：domain=87h, ratio=0.5 → general需要87h (总共174h，各占50%)
    target_general_size = int(len(domain_data) * general_ratio / (1 - general_ratio))

    # 从通用数据中随机采样
    if target_general_size >= len(general_data):
        # 如果需要的量大于可用量，全部使用
        sampled_general = general_data.copy()
        print(
            f"    Warning: Need {target_general_size} general samples, but only {len(general_data)} available"
        )
    else:
        sampled_general = random.sample(general_data, target_general_size)

    # 混合并打乱
    mixed = sampled_general + domain_data
    random.shuffle(mixed)

    actual_ratio = len(sampled_general) / len(mixed)
    print(
        f"    Mixed: {len(sampled_general)} general + {len(domain_data)} domain = {len(mixed)} total"
    )
    print(f"    Actual ratio: {actual_ratio:.1%} general / {1-actual_ratio:.1%} domain")

    return mixed


def main():
    parser = argparse.ArgumentParser(description="准备三阶段训练数据")
    parser.add_argument(
        "--general_train", type=str, required=True, help="通用训练数据路径 (jsonl)"
    )
    parser.add_argument(
        "--general_val", type=str, required=True, help="通用验证数据路径 (jsonl)"
    )
    parser.add_argument(
        "--domain_train", type=str, required=True, help="专业训练数据路径 (jsonl)"
    )
    parser.add_argument(
        "--domain_val", type=str, required=True, help="专业验证数据路径 (jsonl)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/staged", help="输出目录"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    print("=" * 60)
    print("FunASR 三阶段数据准备")
    print("=" * 60)

    # 加载数据
    print("[1/4] Loading data...")
    general_train = load_jsonl(args.general_train)
    general_val = load_jsonl(args.general_val)
    domain_train = load_jsonl(args.domain_train)
    domain_val = load_jsonl(args.domain_val)

    print(f"  General train: {len(general_train)} samples")
    print(f"  General val: {len(general_val)} samples")
    print(f"  Domain train: {len(domain_train)} samples")
    print(f"  Domain val: {len(domain_val)} samples")

    # 阶段1: 50/50 混合
    print("[2/4] Creating Stage 1 data (50% general + 50% domain)...")
    stage1_train = mix_datasets(general_train, domain_train, 0.5)
    stage1_val = mix_datasets(general_val, domain_val, 0.5)
    save_jsonl(stage1_train, f"{args.output_dir}/stage1/train.jsonl")
    save_jsonl(stage1_val, f"{args.output_dir}/stage1/val.jsonl")

    # 阶段2: 20/80 混合
    print("[3/4] Creating Stage 2 data (20% general + 80% domain)...")
    stage2_train = mix_datasets(general_train, domain_train, 0.2)
    stage2_val = mix_datasets(general_val, domain_val, 0.2)
    save_jsonl(stage2_train, f"{args.output_dir}/stage2/train.jsonl")
    save_jsonl(stage2_val, f"{args.output_dir}/stage2/val.jsonl")

    # 阶段3: 纯专业数据
    print("[4/4] Creating Stage 3 data (100% domain)...")
    save_jsonl(domain_train, f"{args.output_dir}/stage3/train.jsonl")
    save_jsonl(domain_val, f"{args.output_dir}/stage3/val.jsonl")

    print("=" * 60)
    print("✓ Data preparation completed!")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print("Next steps:")
    print("  nohup bash auto_finetune.sh > full_train.log 2>&1 &")


if __name__ == "__main__":
    main()
