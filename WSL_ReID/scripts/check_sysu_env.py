#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/kaggle/working/VIREID_Dataset", type=str)
    args = parser.parse_args()
    root = Path(args.data_root) / "SYSU-MM01"

    print("=== PyTorch / GPU ===")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("gpu count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"gpu {i}:", torch.cuda.get_device_name(i))

    print("\n=== SYSU-MM01 layout ===")
    required = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "exp"]
    required += [
        "train_rgb_modified_img.npy",
        "train_rgb_info.npy",
        "train_ir_modified_img.npy",
        "train_ir_info.npy",
        "exp/train_id.txt",
        "exp/val_id.txt",
        "exp/test_id.txt",
    ]
    for rel in required:
        p = root / rel
        print(p, "OK" if p.exists() else "MISSING")
        if not p.exists():
            raise FileNotFoundError(p)

    for name in ["train_rgb_info.npy", "train_ir_info.npy", "train_rgb_modified_img.npy", "train_ir_modified_img.npy"]:
        arr = np.load(root / name, mmap_mode="r")
        print(name, arr.shape, arr.dtype)

    print("\nSYSU environment OK.")


if __name__ == "__main__":
    main()
