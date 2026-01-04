import argparse
import os

import torch

from model import FunASRNano


def main():
    parser = argparse.ArgumentParser(description="FunASR-Nano")
    parser.add_argument("--scp-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument(
        "--model-dir", type=str, default="FunAudioLLM/Fun-ASR-Nano-2512"
    )
    args = parser.parse_args()

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    m, kwargs = FunASRNano.from_pretrained(model=args.model_dir, device=device)
    m.eval()

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(args.scp_file, "r", encoding="utf-8") as f1:
        with open(args.output_file, "w", encoding="utf-8") as f2:
            for line in f1:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    text = m.inference(data_in=[parts[1]], **kwargs)[0][0]["text"]
                    f2.write(f"{parts[0]}\t{text}\n")


if __name__ == "__main__":
    main()
