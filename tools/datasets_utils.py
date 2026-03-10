import json
import os
import wave
from typing import Optional, List, Dict
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 配置日志，方便查看转换进度和错误
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_wav_scp(input_txt_path, output_scp_path, audio_prefix):
    """
    核心函数：生成wav.scp文件
    Args:
        input_txt_path (str): 输入txt文件路径
        output_scp_path (str): 输出wav.scp文件路径
        audio_prefix (str): 音频文件路径前缀
    """
    # 处理路径前缀，确保结尾有且仅有一个 '/'
    if audio_prefix:
        audio_prefix = os.path.join(audio_prefix, "")  # 自动添加路径分隔符

    # 读取输入txt文件并生成内容
    scp_lines = []
    try:
        with open(input_txt_path, "r", encoding="utf-8") as f_in:
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                # 分割每行数据（按制表符或空格分割，兼容不同分隔符）
                parts = line.split(maxsplit=1)  # 只分割第一个空白符
                if len(parts) < 1:
                    logging.warning(
                        f"警告：第{line_num}行格式异常，跳过 -> {line}", file=sys.stderr
                    )
                    continue

                en_chunk = parts[0]
                # 生成scp行（使用两个空格分隔，符合Kaldi标准格式）
                wav_path = f"{audio_prefix}{en_chunk}.wav"
                scp_lines.append(f"{en_chunk}  {wav_path}")

        # 写入输出文件
        with open(output_scp_path, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(scp_lines))

        logging.info(f"✅ 生成成功！")
        logging.info(f"📥 输入文件：{input_txt_path}")
        logging.info(f"📤 输出文件：{output_scp_path}")
        logging.info(f"📊 共处理 {len(scp_lines)} 条记录")
        if audio_prefix:
            logging.info(f"🗂️  音频路径前缀：{audio_prefix}")

    except FileNotFoundError:
        logging.error(f"❌ 错误：输入文件不存在 -> {input_txt_path}", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        logging.error(f"❌ 错误：没有文件读写权限", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ 错误：{str(e)}", file=sys.stderr)
        sys.exit(1)


def jsonl2whisper_dataset(
    input_jsonl: str,
    audio_path_prefix: str,
    output_json: str = r"./dataset.json",
    default_language: str = "en",
    batch_size: int = 1000,
) -> None:
    """
    将JSONL文件转换为Whisper数据集格式（JSON）

    Args:
        input_jsonl: 输入JSONL文件路径
        audio_path: 音频文件存放路径
        output_json: 输出JSON文件路径（包含文件名）
        default_language: 默认语言标签（如"en"、"zh"）
        batch_size: 批量写入阈值（避免内存占用过大）
    """
    # 确保输出目录存在
    output_dir = Path(output_json).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 验证输入文件是否存在
    if not Path(input_jsonl).exists():
        logging.error(f"输入文件不存在: {input_jsonl}")
        raise FileNotFoundError(f"输入文件不存在: {input_jsonl}")

    res_list: List[Dict] = []
    total_count = 0

    try:
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    # 解析JSON行
                    data = json.loads(line)

                    # 提取必要字段（支持从data中读取language，没有则用默认值）
                    source = data.get("source")
                    sentence = data.get("target")
                    language = data.get("language", default_language)

                    # 验证必要字段
                    if not source:
                        logging.warning(f"第{line_num}行缺少必要字段{source}，跳过")
                        continue
                    if not sentence:
                        logging.warning(f"第{line_num}行缺少必要字段{sentence}，跳过")
                        continue
                    # 转换为绝对路径（Whisper更可靠）
                    fileName = Path(source).name
                    audio_path = Path(audio_path_prefix).resolve() / fileName
                    if not audio_path.exists():
                        logging.warning(
                            f"第{line_num}行音频文件不存在: {audio_path}，跳过"
                        )
                        continue

                    # 获取音频时长（带异常处理）
                    duration = _get_wav_duration(str(audio_path))
                    if duration is None:
                        logging.warning(
                            f"第{line_num}行获取音频时长失败: {audio_path}，跳过"
                        )
                        continue

                    # 构建Whisper格式数据
                    res = _build_whisper_entity(source, sentence, language, duration)
                    res_list.append(res)
                    total_count += 1

                    # 批量写入
                    if len(res_list) >= batch_size:
                        write_batch(
                            output_json, res_list, append=os.path.exists(output_json)
                        )
                        logging.info(f"已处理 {total_count} 条数据，批量写入完成")
                        res_list.clear()

                except json.JSONDecodeError:
                    logging.error(f"第{line_num}行JSON格式错误，跳过")
                    continue
                except Exception as e:
                    logging.error(f"第{line_num}行处理失败: {str(e)}，跳过")
                    continue

        # 写入剩余数据
        if res_list:
            write_batch(output_json, res_list, append=os.path.exists(output_json))
            logging.info(f"处理完成，最后一批 {len(res_list)} 条数据写入完成")

        logging.info(
            f"转换成功！共处理 {total_count} 条有效数据，输出文件: {output_json}"
        )

    except Exception as e:
        logging.error(f"转换过程中发生严重错误: {str(e)}")
        raise


def _build_whisper_entity(
    audio_path: str, sentence: str, language: str, duration: float
) -> Dict:
    """构建Whisper数据集的单条样本格式"""
    return {
        "audio": {"path": audio_path},
        "sentence": sentence.strip(),  # 去除前后空格
        "language": language.lower(),  # 统一转为小写
        "duration": duration,
    }


# 获取单个WAV文件时长
def _get_wav_duration(file_path: str) -> Optional[float]:
    """
    获取WAV文件时长（秒），支持异常处理

    Args:
        file_path: WAV文件绝对路径

    Returns:
        时长（保留2位小数），失败返回None
    """
    try:
        with wave.open(file_path, "r") as wav_file:
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            duration = n_frames / float(frame_rate)
            return round(duration, 2)
    except wave.Error:
        logging.warning(f"文件不是有效的WAV格式: {file_path}")
        return None
    except PermissionError:
        logging.warning(f"没有权限访问文件: {file_path}")
        return None
    except Exception as e:
        logging.warning(f"获取音频时长失败: {str(e)}")
        return None


def get_total_wav_duration(wav_dir: str) -> float:
    import soundfile as sf

    total_seconds = 0.0

    for root, _, files in os.walk(wav_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                path = os.path.join(root, file)
                info = sf.info(path)
                total_seconds += info.frames / info.samplerate

    return total_seconds


# 批量写入JSON数组
def write_batch(output_path: str, data_list: List[Dict], append: bool = False) -> None:
    """
    批量写入数据到JSON文件，保证整个文件只有一个数组括号

    Args:
        output_path: 输出文件路径
        data_list: 要写入的数据列表
        append: 是否追加模式（首次写入用False，后续批量用True）
    """
    # 生成当前批次的文本（每条按 pretty 格式，但批次内以逗号分隔）
    items = []
    for item in data_list:
        items.append(json.dumps(item, ensure_ascii=False, indent=2))
    batch_text = ",\n".join(items)

    # 如果不追加或文件不存在/为空，则直接写入完整数组
    if (
        not append
        or not os.path.exists(output_path)
        or os.path.getsize(output_path) == 0
    ):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("[\n")
            f.write(batch_text)
            f.write("\n]")
        return

    # 追加到已有的 JSON 数组：读取原文件、去掉末尾的']'，再写回并追加新批次，最后补上']'
    with open(output_path, "r+", encoding="utf-8") as f:
        content = f.read()
        if not content.strip():
            # 文件为空的情形，直接写新数组
            f.seek(0)
            f.write("[\n")
            f.write(batch_text)
            f.write("\n]")
            f.truncate()
            return

        # 去掉末尾的 ']' 以及多余的逗号/空白
        stripped = content.rstrip()
        if stripped.endswith("]"):
            stripped = stripped[: stripped.rfind("]")].rstrip()
        stripped = stripped.rstrip(",\n")

        # 如果原文件只有'['（即还没有元素），直接追加元素
        if stripped.endswith("["):
            new_content = stripped + "\n" + batch_text + "\n]"
        else:
            new_content = stripped + ",\n" + batch_text + "\n]"

        f.seek(0)
        f.write(new_content)
        f.truncate()


# json压缩
def json_compress(input_json: str, output_json: str) -> None:
    """将JSON文件压缩为单行格式，节省空间"""
    with open(input_json, "r", encoding="utf-8") as f_in:
        data = json.load(f_in)
    with open(output_json, "w", encoding="utf-8") as f_out:
        json.dump(data, f_out, ensure_ascii=False, separators=(",", ":"))


# json转jsonl
def json2jsonl(input_json: str, output_jsonl: str) -> None:
    """将JSON文件转换为JSONL格式"""
    with open(input_json, "r", encoding="utf-8") as f_in:
        data = json.load(f_in)
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for item in data:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")


# excel转jsonl
def excel2jsonl(
    input_excel: str,
    output_jsonl: str,
    audio_path_prefix: Optional[str] = None,
    default_language: str = "en",
) -> None:
    """将Excel文件转换为Whisper JSONL格式（适用于只有 path 和 sentence 列的表格）"""
    import pandas as pd

    # 只支持 path 和 sentence 两列（如果列名不同，请在调用前重命名）
    audio_keys = ["path"]
    text_keys = ["sentence"]
    lang_keys = ["language", "lang"]

    # 读取表格
    df = pd.read_excel(input_excel, dtype=str).fillna("")

    # 确保输出目录存在
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for idx, row in df.iterrows():
            try:
                # 简单字段选择器（只查找 path 和 sentence）
                def pick(keys):
                    for k in keys:
                        if k in row.index and row[k] and str(row[k]).strip():
                            return str(row[k]).strip()
                    return None

                src = pick(audio_keys)
                sentence = pick(text_keys)
                language = pick(lang_keys) or default_language

                if not src:
                    logging.warning(f"第{idx+1}行缺少 path 字段，跳过")
                    continue
                if not sentence:
                    logging.warning(f"第{idx+1}行缺少 sentence 字段，跳过")
                    continue

                # 解析并定位音频文件：若提供前缀则用前缀+文件名；否则尝试原始路径或绝对化
                src_path = Path(src)
                if audio_path_prefix:
                    # 如果 src 看起来像仅文件名（无目录分隔符），使用前缀 + 名称
                    if not any(sep in src for sep in (os.sep, "/")):
                        audio_path = Path(audio_path_prefix).resolve() / src_path.name
                    else:
                        # 若 src 包含路径但不是绝对路径，先尝试相对于前缀
                        if not src_path.is_absolute():
                            candidate = Path(audio_path_prefix).resolve() / src_path
                            audio_path = (
                                candidate if candidate.exists() else src_path.resolve()
                            )
                        else:
                            audio_path = src_path
                else:
                    audio_path = (
                        src_path if src_path.is_absolute() else src_path.resolve()
                    )

                if not audio_path.exists():
                    logging.warning(f"第{idx+1}行音频文件不存在: {audio_path}，跳过")
                    continue

                duration = _get_wav_duration(str(audio_path))
                if duration is None:
                    logging.warning(f"第{idx+1}行无法获取音频时长: {audio_path}，跳过")
                    continue

                # 构建并写入 Whisper 格式实体（audio.path 使用绝对路径）
                entity = _build_whisper_entity(
                    str(audio_path), sentence, language, duration
                )
                f_out.write(json.dumps(entity, ensure_ascii=False) + "\n")
                written += 1

            except Exception as e:
                logging.error(f"第{idx+1}行处理失败: {e}, 跳过")
                continue

    logging.info(f"已写入 {written} 条 Whisper JSONL 记录 到 {output_jsonl}")


# csv转text
def csv2text(csv_file: str):
    import csv

    # 获取 CSV 文件所在目录和文件名（不含后缀）
    dir_name = os.path.dirname(csv_file)
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    # 构造 TXT 文件路径
    txt_file = os.path.join(dir_name, base_name + ".txt")
    # 转换 CSV → TXT
    with (
        open(csv_file, "r", encoding="utf-8") as f_in,
        open(txt_file, "w", encoding="utf-8") as f_out,
    ):
        next(f_in)  # 跳过表头
        for line in f_in:
            line = line.strip()
            if line:
                file_path, text = line.split(",", 1)
                file_name = file_path.split("/")[-1].removesuffix(
                    ".wav"
                )  # 取最后一个 / 之后的文件名,并去掉.wav后缀
                f_out.write(f"{file_name} {text}\n")

    print(f"转换完成，TXT 文件路径：{txt_file}")


# 把子目录文件内容，提到顶级父目录下
def flatten_subdirs_to_root(root_dir: str, remove_empty_dirs: bool = True):
    """
    遍历 root_dir 下的所有子目录，把文件移动到顶层目录。

    参数:
    - root_dir: 顶层父目录路径
    - remove_empty_dirs: 是否删除移动后空的子目录，默认 True
    """
    import shutil

    root_dir = os.path.abspath(root_dir)  # 转绝对路径

    # 遍历所有子目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath == root_dir:
            continue  # 跳过顶层目录
        for file in filenames:
            src_path = os.path.join(dirpath, file)
            dst_path = os.path.join(root_dir, file)

            # 避免同名文件覆盖
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(file)
                count = 1
                while True:
                    new_name = f"{base}_{count}{ext}"
                    dst_path = os.path.join(root_dir, new_name)
                    if not os.path.exists(dst_path):
                        break
                    count += 1

            shutil.move(src_path, dst_path)

    # 删除空子目录
    if remove_empty_dirs:
        for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
            if dirpath != root_dir and not os.listdir(dirpath):
                os.rmdir(dirpath)

    print(f"文件已全部提取到顶层目录: {root_dir}")

def paraformerJsonl2Qwen3ASRJsonl(input_file: str, output_file: str):
    """
    基于生成的paraformer输入格式转为qwen3-asr输入格式
    """
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            source = data["source"]
            target = data["target"]
            if source.startswith("data-en"):
                lang = "English"
            elif source.startswith("data-zh"):
                lang = "Chinese"
            else:
                lang = "None"
            out = {"audio": source, "text": f"language {lang}<asr_text>{target}"}
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    from pathlib import Path

    target_dir = Path(r"data/staged/stage1")
    # 获取目录下音频总时长
    # input_dir = target_dir / "audio_dir"
    # print(get_total_wav_duration(str(input_dir)))

    # 生成txt文件
    # csv2text(str(target_dir / "wav.csv"))

    # 生成scp文件
    # input_txt_path = target_dir / "wav.txt"
    # generate_wav_scp(
    #     input_txt_path,
    #     output_scp_path=str(input_txt_path.with_suffix(".scp")),
    #     audio_prefix=str(target_dir / "wav"),
    # )

    # flatten_subdirs_to_root(r"data/general/train/wav")

    # paraformer输入数据集转为qwen3-asr
    paraformerJsonl2Qwen3ASRJsonl(str(target_dir / "train_paraformer.jsonl"),str(target_dir / "train_qwen3asr.jsonl"))
    # JSONL转Whisper数据集
    try:
        input = target_dir / "dataset.json"
        # jsonl2whisper_dataset(
        #     input_jsonl=str(target_dir),
        #     audio_path_prefix=str(target_dir / "wav"),
        # )

        # json_compress(
        #     input_json=r"./dataset.json",
        #     output_json=r"./dataset.compress.json",
        # )

        # json2jsonl(
        #     input_json=input,
        #     output_jsonl=r"./dataset.jsonl",
        # )

        # excel2jsonl(
        #     input_excel=str(target_dir / "test.xlsx"),
        #     output_jsonl=r"./dataset_from_excel.jsonl",
        #     audio_path_prefix=str(target_dir / "wav"),
        # )
    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}")
        exit(1)
