# 使用官方 python 镜像作为基础镜像
FROM python:3.11-slim AS base
LABEL maintainer="YuanJie" \
    description="An Acoustic Feature Detection Mirror Construction Project" \
    license="MIT" \
    email="wangjh0825@qq.com"
# 接收 flash_attn 参数
ARG FLASH_ATTN
# 设置非交互模式，避免 apt 安装时卡住
ENV DEBIAN_FRONTEND=noninteractive
# 写入阿里云 Debian 12 源（deb822 格式）
RUN cat > /etc/apt/sources.list.d/debian.sources <<'EOF'
Types: deb
URIs: https://mirrors.aliyun.com/debian
Suites: bookworm bookworm-updates
Components: main contrib non-free non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg

Types: deb
URIs: https://mirrors.aliyun.com/debian-security
Suites: bookworm-security
Components: main contrib non-free non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
EOF

# 系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg\
    curl \
    procps \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 用 PyPI 国内源安装 uv（稳定）
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# 增加 uv 下载超时（单位秒，建议 300+）
ENV UV_HTTP_TIMEOUT=600
# 用 pip3 安装 uv
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -U uv -i ${UV_INDEX_URL} && \
    uv --version

ENV PATH="/root/.local/bin:${PATH}"

# 设置工作目录
WORKDIR /workspace

# 拷贝依赖文件（先拷贝依赖有利于 Docker 层缓存）
COPY pyproject.toml .python-version ./

# 同步依赖，--active 强制使用当前 venv，避免重建
RUN uv sync --extra cu128 --active
RUN uv pip install transformers==4.57.6 peft funasr==1.3.1

# 判断是否为空，只有非空才安装 flash-attn
RUN if [ -n "$FLASH_ATTN" ]; then \
    echo "Installing flash-attn..."; \
    uv pip install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl; \
    uv pip install flash-attn datasets qwen_asr; \
    else \
    echo "Skipping flash-attn installation"; \
    fi
# 再拷贝项目代码
COPY . .

# 给 run.sh 可执行权限
RUN chmod +x auto_finetune.sh finetune_nano.sh finetune_paraformer.sh finetune_qwen3asr.sh

# uv 会创建 .venv，这里将其添加到 PATH
ENV PATH="/workspace/.venv/bin:${PATH}"

# 不兜底
CMD ["/bin/bash"]