#!/usr/bin/env python3
"""Check Kaggle GPU and RegDB layout before training."""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/kaggle/working/VIREID_Dataset")
    args = parser.parse_args()

    print("=== PyTorch / GPU ===")
    try:
        import torch
        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        print("gpu count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"gpu {i}:", torch.cuda.get_device_name(i))
    except Exception as exc:
        print("torch check failed:", repr(exc))

    print("\n=== RegDB layout ===")
    base = Path(args.data_root) / "RegDB"
    required = [
        base / "idx/train_visible_1.txt",
        base / "idx/train_thermal_1.txt",
        base / "idx/test_visible_1.txt",
        base / "idx/test_thermal_1.txt",
    ]
    for f in required:
        print(f, "OK" if f.exists() else "MISSING")
        if not f.exists():
            raise FileNotFoundError(f)

    for f in required:
        first = f.read_text().splitlines()[0]
        rel, label = first.split()[:2]
        img_path = base / rel
        print("\n", f.name)
        print("first line:", first)
        print("image:", img_path)
        print("exists:", img_path.exists())
        if not img_path.exists():
            raise FileNotFoundError(img_path)
        img = Image.open(img_path)
        print("image size:", img.size, "mode:", img.mode, "label:", label)

    print("\nEnvironment and RegDB layout OK.")


if __name__ == "__main__":
    main()
