#!/usr/bin/env python3
"""
绘制训练曲线
用法: python monitor_log.py outputs/stage3_finetune/log.txt
"""

import re
import sys
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log(log_file):
    """解析训练日志"""
    train_losses = []
    valid_losses = []
    valid_accs = []
    epochs = []

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 提取 epoch 信息
            epoch_match = re.search(r"epoch[:\s]+(\d+)", line, re.IGNORECASE)

            # 匹配训练损失
            if "train_loss" in line.lower():
                match = re.search(r"train_loss[:\s]+([0-9.]+)", line, re.IGNORECASE)
                if match:
                    train_losses.append(float(match.group(1)))
                    if epoch_match:
                        epochs.append(int(epoch_match.group(1)))

            # 匹配验证损失
            if "valid_loss" in line.lower():
                match = re.search(r"valid_loss[:\s]+([0-9.]+)", line, re.IGNORECASE)
                if match:
                    valid_losses.append(float(match.group(1)))

            # 匹配验证准确率
            if "valid_acc" in line.lower():
                match = re.search(r"valid_acc[:\s]+([0-9.]+)", line, re.IGNORECASE)
                if match:
                    valid_accs.append(float(match.group(1)))

    return train_losses, valid_losses, valid_accs


def plot_curves(train_losses, valid_losses, valid_accs, output_dir):
    """绘制训练曲线"""

    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 子图1: 损失曲线
    ax1 = axes[0]
    if train_losses:
        ax1.plot(train_losses, label="Train Loss", color="blue", alpha=0.7, linewidth=2)
    if valid_losses:
        # 验证损失通常比训练步少，需要调整x轴
        valid_x = [
            i * (len(train_losses) // len(valid_losses))
            for i in range(len(valid_losses))
        ]
        ax1.plot(
            valid_x,
            valid_losses,
            label="Valid Loss",
            color="red",
            alpha=0.7,
            linewidth=2,
            marker="o",
        )

    ax1.set_xlabel("Steps", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 标注过拟合区域
    if train_losses and valid_losses and len(valid_losses) >= 2:
        # 找到验证损失开始上升的点
        for i in range(1, len(valid_losses)):
            if valid_losses[i] > valid_losses[i - 1]:
                x_pos = i * (len(train_losses) // len(valid_losses))
                ax1.axvline(
                    x=x_pos, color="orange", linestyle="--", alpha=0.5, linewidth=1.5
                )
                ax1.text(
                    x_pos,
                    max(valid_losses),
                    " ⚠️ 可能过拟合",
                    fontsize=9,
                    color="orange",
                    verticalalignment="top",
                )
                break

    # 子图2: 验证准确率
    ax2 = axes[1]
    if valid_accs:
        valid_x = [
            i * (len(train_losses) // len(valid_accs)) for i in range(len(valid_accs))
        ]
        ax2.plot(
            valid_x,
            valid_accs,
            label="Valid Accuracy",
            color="green",
            alpha=0.7,
            linewidth=2,
            marker="s",
        )

        # 标注最高点
        max_acc_idx = valid_accs.index(max(valid_accs))
        max_acc_x = max_acc_idx * (len(train_losses) // len(valid_accs))
        ax2.axvline(x=max_acc_x, color="green", linestyle="--", alpha=0.5)
        ax2.text(
            max_acc_x,
            max(valid_accs),
            f" ✓ Best: {max(valid_accs):.4f}",
            fontsize=10,
            color="green",
            verticalalignment="bottom",
        )

    ax2.set_xlabel("Steps", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    output_path = Path(output_dir) / "training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ 图表已保存: {output_path}")

    # 显示图片
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("用法: python plot_training_curves.py <log_file>")
        print("示例: python plot_training_curves.py outputs/stage3_finetune/log.txt")
        sys.exit(1)

    log_file = sys.argv[1]
    output_dir = Path(log_file).parent

    print(f"📖 读取日志: {log_file}")

    try:
        train_losses, valid_losses, valid_accs = parse_log(log_file)
    except FileNotFoundError:
        print(f"❌ 文件不存在: {log_file}")
        sys.exit(1)

    if not train_losses and not valid_losses:
        print("❌ 日志中未找到损失数据")
        sys.exit(1)

    print(f"📊 找到数据:")
    print(f"  训练损失: {len(train_losses)} 条")
    print(f"  验证损失: {len(valid_losses)} 条")
    print(f"  验证准确率: {len(valid_accs)} 条")

    plot_curves(train_losses, valid_losses, valid_accs, output_dir)


if __name__ == "__main__":
    main()
