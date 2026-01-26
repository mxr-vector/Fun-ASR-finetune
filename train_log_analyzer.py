# =========================
# App Bootstrap (Qt + Matplotlib)
# =========================
import os
import sys
import re
from dataclasses import dataclass
from typing import Optional, List

# ---------- UTF-8 兜底 ----------
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ---------- Qt 高 DPI ----------
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QComboBox,
    QHeaderView,
    QSplitter,
    QGroupBox,
    QTabWidget,
)
from PyQt6.QtGui import QFont, QFontDatabase, QColor

QApplication.setHighDpiScaleFactorRoundingPolicy(
    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
)

# ---------- matplotlib ----------
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt


# =========================
# 中文字体设置
# =========================
def setup_qt_chinese_font(app: QApplication):
    preferred_fonts = [
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
    ]

    available_fonts = set(QFontDatabase.families())
    for font in preferred_fonts:
        if font in available_fonts:
            app.setFont(QFont(font, 10))
            print(f"[Qt] 使用字体: {font}")
            return

    print("[Qt] ⚠ 未检测到中文字体，使用系统默认")


# =========================
# 数据结构
# =========================
@dataclass
class TrainRecord:
    phase: str
    rank: int
    epoch: int
    epoch_total: int
    step: int
    total_step: int
    loss: float
    acc: float
    lr: float
    total_time: float
    gpu_mem: float


# =========================
# 日志解析正则
# =========================
LOG_PATTERN = re.compile(
    r"(?P<phase>train|val), rank: (?P<rank>\d+), "
    r"epoch: (?P<epoch>\d+)/(?P<epoch_total>\d+).*?"
    r"total step: (?P<total_step>\d+).*?"
    r"\(loss_avg_rank: (?P<loss>[0-9.eE+-]+)\).*?"
    r"\(acc_avg_slice: (?P<acc>[0-9.eE+-]+)\).*?"
    r"\(lr: (?P<lr>[0-9.eE+-]+)\).*?"
    r"\{.*?total_time\': \'(?P<time>[0-9.eE+-]+)\'\}.*?"
    r"usage: (?P<gpu>[0-9.]+) GB"
)


def parse_log_line(line: str) -> Optional[TrainRecord]:
    m = LOG_PATTERN.search(line)
    if not m:
        return None

    return TrainRecord(
        phase=m.group("phase"),
        rank=int(m.group("rank")),
        epoch=int(m.group("epoch")),
        epoch_total=int(m.group("epoch_total")),
        step=0,
        total_step=int(m.group("total_step")),
        loss=float(m.group("loss")),
        acc=float(m.group("acc")),
        lr=float(m.group("lr")),
        total_time=float(m.group("time")),
        gpu_mem=float(m.group("gpu")),
    )


# =========================
# 主窗口
# =========================
class TrainLogAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("训练日志分析器 / Training Log Analyzer")
        self.resize(1600, 900)

        self.records: List[TrainRecord] = []

        # ===== 统计面板 =====
        self.stats_group = QGroupBox("训练统计")
        stats_layout = QHBoxLayout()

        self.stat_labels = {
            "total": QLabel("总记录: 0"),
            "avg_loss": QLabel("平均Loss: -"),
            "best_acc": QLabel("最佳Acc: -"),
            "avg_gpu": QLabel("平均显存: -"),
            "avg_time": QLabel("平均耗时: -"),
        }

        for label in self.stat_labels.values():
            label.setStyleSheet("font-weight: bold; padding: 5px;")
            stats_layout.addWidget(label)

        self.stats_group.setLayout(stats_layout)

        # ===== 标签页控件 =====
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                background: white;
            }
            QTabBar::tab {
                background: #f0f0f0;
                padding: 10px 20px;
                margin-right: 2px;
                border: 1px solid #d0d0d0;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: white;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background: #e5e5e5;
            }
        """
        )

        # ===== 图表标签页 =====
        chart_tab = QWidget()
        chart_layout = QVBoxLayout()

        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)

        # 创建2x2子图
        self.axes = {
            "loss": self.figure.add_subplot(2, 2, 1),
            "acc": self.figure.add_subplot(2, 2, 2),
            "lr": self.figure.add_subplot(2, 2, 3),
            "gpu": self.figure.add_subplot(2, 2, 4),
        }

        self.figure.tight_layout(pad=3.0)
        chart_layout.addWidget(self.canvas)
        chart_tab.setLayout(chart_layout)

        # ===== 表格标签页 =====
        table_tab = QWidget()
        table_layout = QVBoxLayout()

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            [
                "阶段",
                "Rank",
                "Epoch",
                "Total Step",
                "Loss",
                "Acc",
                "LR",
                "Step Time(s)",
                "GPU(GB)",
                "状态",
            ]
        )

        # 表格优化
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            """
            QTableWidget {
                gridline-color: #d0d0d0;
                selection-background-color: #0078d4;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 5px;
                border: 1px solid #d0d0d0;
                font-weight: bold;
            }
        """
        )

        # 自适应列宽
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(9, QHeaderView.ResizeMode.Stretch)

        table_layout.addWidget(self.table)
        table_tab.setLayout(table_layout)

        # ===== 添加标签页 =====
        self.tab_widget.addTab(chart_tab, "图表视图")
        self.tab_widget.addTab(table_tab, "表格视图")

        # ===== 控件区 =====
        control_layout = QHBoxLayout()

        open_btn = QPushButton("打开训练日志")
        open_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """
        )
        open_btn.clicked.connect(self.load_log)

        export_btn = QPushButton("导出图表")
        export_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #107c10;
                color: white;
                border: none;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0e6b0e;
            }
        """
        )
        export_btn.clicked.connect(self.export_plot)

        self.info_label = QLabel("加载日志后将自动分析训练状态")
        self.info_label.setStyleSheet("color: #666; font-style: italic;")

        control_layout.addWidget(open_btn)
        control_layout.addWidget(export_btn)
        control_layout.addStretch()
        control_layout.addWidget(self.info_label)

        # ===== 主布局 =====
        main_layout = QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.stats_group)
        main_layout.addWidget(self.tab_widget)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # =========================
    # 加载日志
    # =========================
    def load_log(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择训练日志文件", "", "Log Files (*.log *.txt);;All Files (*)"
        )
        if not path:
            return

        self.records.clear()
        self.table.setRowCount(0)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                rec = parse_log_line(line)
                if rec:
                    self.records.append(rec)
                    self.add_record(rec)

        self.update_statistics()
        self.update_plots()
        self.info_label.setText(f"已解析 {len(self.records)} 条训练记录")

    # =========================
    # 更新统计信息
    # =========================
    def update_statistics(self):
        if not self.records:
            return

        train_recs = [r for r in self.records if r.phase == "train"]

        avg_loss = (
            sum(r.loss for r in train_recs) / len(train_recs) if train_recs else 0
        )
        best_acc = max((r.acc for r in self.records), default=0)
        avg_gpu = sum(r.gpu_mem for r in self.records) / len(self.records)
        avg_time = sum(r.total_time for r in self.records) / len(self.records)

        self.stat_labels["total"].setText(f"总记录: {len(self.records)}")
        self.stat_labels["avg_loss"].setText(f"平均Loss: {avg_loss:.4f}")
        self.stat_labels["best_acc"].setText(f"最佳Acc: {best_acc:.3f}")
        self.stat_labels["avg_gpu"].setText(f"平均显存: {avg_gpu:.2f} GB")
        self.stat_labels["avg_time"].setText(f"平均耗时: {avg_time:.3f}s")

    # =========================
    # 表格填充
    # =========================
    def add_record(self, r: TrainRecord):
        row = self.table.rowCount()
        self.table.insertRow(row)

        status = "✓ OK"
        status_color = QColor(0, 128, 0)

        if r.loss > 1.0:
            status = "⚠ Loss偏高"
            status_color = QColor(255, 140, 0)
        if r.gpu_mem > 20:
            status = "⚠ 显存偏高"
            status_color = QColor(255, 0, 0)
        if r.total_time > 1.0:
            status = "⚠ Step偏慢"
            status_color = QColor(255, 140, 0)

        values = [
            r.phase,
            r.rank,
            f"{r.epoch}/{r.epoch_total}",
            r.total_step,
            f"{r.loss:.4f}",
            f"{r.acc:.3f}",
            f"{r.lr:.2e}",
            f"{r.total_time:.3f}",
            f"{r.gpu_mem:.2f}",
            status,
        ]

        for col, v in enumerate(values):
            item = QTableWidgetItem(str(v))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            if col == 9:  # 状态列
                item.setForeground(status_color)

            self.table.setItem(row, col, item)

    # =========================
    # 图表更新（2x2网格）
    # =========================
    def update_plots(self):
        for ax in self.axes.values():
            ax.clear()

        if not self.records:
            self.canvas.draw()
            return

        # 配置
        configs = [
            ("loss", "Loss", self.axes["loss"]),
            ("acc", "Accuracy", self.axes["acc"]),
            ("lr", "Learning Rate", self.axes["lr"]),
            ("gpu", "GPU Memory (GB)", self.axes["gpu"]),
        ]

        for metric_key, title, ax in configs:
            for phase, color in [
                ("train", "#1f77b4"),
                ("val", "#ff7f0e"),
            ]:
                phase_recs = [r for r in self.records if r.phase == phase]
                if not phase_recs:
                    continue

                xs = [r.total_step for r in phase_recs]

                if metric_key == "loss":
                    ys = [r.loss for r in phase_recs]
                elif metric_key == "acc":
                    ys = [r.acc for r in phase_recs]
                elif metric_key == "lr":
                    ys = [r.lr for r in phase_recs]
                else:  # gpu
                    ys = [r.gpu_mem for r in phase_recs]

                ax.plot(xs, ys, label=phase, color=color, linewidth=1.0, alpha=0.9)

            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Total Step", fontsize=9)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(labelsize=8)

        self.figure.tight_layout(pad=2.5)
        self.canvas.draw()

    # =========================
    # 导出图表
    # =========================
    def export_plot(self):
        if not self.records:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图表",
            "training_analysis.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if path:
            self.figure.savefig(path, dpi=300, bbox_inches="tight")
            self.info_label.setText(f"图表已保存到: {path}")


# =========================
# 程序入口
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    setup_qt_chinese_font(app)

    win = TrainLogAnalyzer()
    win.show()

    sys.exit(app.exec())
