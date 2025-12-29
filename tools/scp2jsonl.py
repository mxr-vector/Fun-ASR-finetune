import argparse
import json
import os
from io import BytesIO
from urllib.request import urlopen

import soundfile as sf
from modelscope import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scp-file", type=str, required=True)
    parser.add_argument("--transcript-file", type=str, required=True)
    parser.add_argument("--jsonl-file", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    scp_file = args.scp_file
    transcript_file = args.transcript_file
    jsonl_file = args.jsonl_file

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    f = open(jsonl_file, "w")
    with open(scp_file, "r") as f1, open(transcript_file, "r") as f2:
        for line1, line2 in zip(f1, f2):
            line1, line2 = line1.strip(), line2.strip()
            if not line1 or not line2:
                continue
            parts1, parts2 = line1.split(maxsplit=1), line2.split(maxsplit=1)
            if len(parts1) != 2 or len(parts2) != 2:
                continue
            utt1, utt2 = parts1[0], parts2[0]
            wav_path, text = parts1[1], parts2[1]
            if utt1 != utt2:
                print(f"UTT mismatch, skip: {utt1} vs {utt2}")
                continue
            # TODO: avoid downloading the total audio file to memory
            if wav_path.startswith("http"):
                response = urlopen(wav_path)
                if response.status != 200:
                    print(f"WAV path not found, skip: {wav_path}")
                    continue
                audio_file = BytesIO(response.read())
                duration = sf.info(audio_file).duration
            else:
                if not os.path.exists(wav_path):
                    print(f"WAV path not found, skip: {wav_path}")
                    continue
                duration = sf.info(wav_path).duration

            data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"语音转写：<|startofspeech|>!{wav_path}<|endofspeech|>"},
                    {"role": "assistant", "content": text}
                ],
                "speech_length": int((duration * 1000 - 25) // 10 + 1),
                "text_length": len(tokenizer.tokenize(text))
            }
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")
    f.close()


if __name__ == "__main__":
    main()
