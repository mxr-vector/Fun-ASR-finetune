import os
from funasr import AutoModel


def main():
    model_dir = "models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

    # 初始化模型
    model = AutoModel(
        model=model_dir,
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        punc_model="ct-punc-c",
        punc_model_revision="v2.0.4",
        # spk_model="cam++", spk_model_revision="v2.0.2",
    )

    test_dir = "data/test"

    results = []  # 统一收集所有结果

    # 遍历目录下所有 WAV 文件
    for file_name in os.listdir(test_dir):
        if file_name.lower().endswith((".wav", ".WAV")):
            wav_path = os.path.join(test_dir, file_name)
            print(f"Processing: {wav_path}")

            try:
                res = model.generate(input=wav_path, batch_size_s=300)
            except Exception as e:
                print(f"Failed to process {wav_path}: {e}")
                continue

            # 统一转为 list 处理
            if isinstance(res, list):
                for r in res:
                    results.append(
                        {
                            "file": file_name,
                            "key": r.get("key", ""),
                            "text": r.get("text", ""),
                        }
                    )
            else:
                results.append(
                    {
                        "file": file_name,
                        "key": res.get("key", ""),
                        "text": res.get("text", ""),
                    }
                )

    # 最后统一输出
    print("\n===== FINAL RESULT =====")
    for item in results:
        print(f"{item['file']} | {item['key']}: {item['text']}")


if __name__ == "__main__":
    main()
