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
    QWidget,
    QPushButton,
    QLabel,
    QComboBox,
)
from PyQt6.QtGui import QFont, QFontDatabase

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
        "Noto Sans CJK SC",  # Linux / WSL
        "Microsoft YaHei",  # Windows
        "SimHei",  # fallback
    ]

    available_fonts = set(QFontDatabase.families())
    for font in preferred_fonts:
        if font in available_fonts:
            app.setFont(QFont(font, 10))
            print(f"[Qt] 使用字体: {font}")
            return

    print("[Qt] ⚠ 未检测到中文字体，使用系统默认")


def setup_matplotlib_chinese_font():
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
    ]
    plt.rcParams["axes.unicode_minus"] = False


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
        self.resize(1400, 800)

        self.records: List[TrainRecord] = []

        # ===== 表格 =====
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
                "GPU 显存(GB)",
                "状态",
            ]
        )

        # ===== 图表 =====
        self.figure = Figure(figsize=(6, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Loss", "Acc", "LR", "Step Time", "GPU Mem"])
        self.metric_combo.currentIndexChanged.connect(self.update_plot)

        # ===== 控件 =====
        self.info_label = QLabel("加载日志后将自动分析训练状态")
        open_btn = QPushButton("打开训练日志")
        open_btn.clicked.connect(self.load_log)

        layout = QVBoxLayout()
        layout.addWidget(open_btn)
        layout.addWidget(self.metric_combo)
        layout.addWidget(self.canvas)
        layout.addWidget(self.info_label)
        layout.addWidget(self.table)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # =========================
    # 加载日志
    # =========================
    def load_log(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择训练日志文件", "", "Log Files (*.log *.txt)"
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

        self.info_label.setText(f"已解析 {len(self.records)} 条训练记录")
        self.update_plot()

    # =========================
    # 表格填充
    # =========================
    def add_record(self, r: TrainRecord):
        row = self.table.rowCount()
        self.table.insertRow(row)

        status = "OK"
        if r.loss > 1.0:
            status = "⚠ Loss 偏高"
        if r.gpu_mem > 20:
            status = "⚠ 显存偏高"
        if r.total_time > 1.0:
            status = "⚠ Step 偏慢"

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
            self.table.setItem(row, col, item)

    # =========================
    # 图表更新
    # =========================
    def update_plot(self):
        self.ax.clear()
        if not self.records:
            return

        metric = self.metric_combo.currentText()

        def get_value(r: TrainRecord):
            return {
                "Loss": r.loss,
                "Acc": r.acc,
                "LR": r.lr,
                "Step Time": r.total_time,
                "GPU Mem": r.gpu_mem,
            }[metric]

        for phase, color in [("train", "tab:blue"), ("val", "tab:orange")]:
            xs = [r.total_step for r in self.records if r.phase == phase]
            ys = [get_value(r) for r in self.records if r.phase == phase]
            if xs:
                self.ax.plot(xs, ys, label=phase, color=color)

        self.ax.set_title(metric)
        self.ax.set_xlabel("Total Step")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()


# =========================
# 程序入口
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    setup_qt_chinese_font(app)
    setup_matplotlib_chinese_font()

    win = TrainLogAnalyzer()
    win.show()

    sys.exit(app.exec())
