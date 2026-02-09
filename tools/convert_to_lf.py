#!/usr/bin/env python3
import os

# 要处理的文件后缀
EXTS = [".py", ".sh", ".md", ".toml"]
# 特殊文件名
FILES = ["Dockerfile"]

for root, dirs, files in os.walk("."):
    for f in files:
        path = os.path.join(root, f)
        if any(f.endswith(ext) for ext in EXTS) or f in FILES:
            with open(path, "rb") as infile:
                content = infile.read()
            # 替换 CRLF 为 LF
            content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            with open(path, "wb") as outfile:
                outfile.write(content)
            print(f"Converted LF: {path}")
