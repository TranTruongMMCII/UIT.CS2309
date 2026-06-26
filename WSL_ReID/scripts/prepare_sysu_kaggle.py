#!/usr/bin/env python3
"""Prepare SYSU-MM01 for WSL-VIReID on Kaggle.

Expected final layout:
  <data-root>/SYSU-MM01/
    cam1/ ... cam6/
    exp/train_id.txt, val_id.txt, test_id.txt
    train_rgb_modified_img.npy
    train_rgb_info.npy
    train_ir_modified_img.npy
    train_ir_info.npy

The official repo uses pre_process_sysu.py with a hard-coded path. This script
is a Kaggle-safe replacement that either links existing preprocessed numpy files
or creates them in /kaggle/working.
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

IMAGE_HEIGHT = 288
IMAGE_WIDTH = 144
RGB_CAMERAS = ["cam1", "cam2", "cam4", "cam5"]
IR_CAMERAS = ["cam3", "cam6"]
REQUIRED_CAMERAS = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6"]
PREPROCESSED_FILES = [
    "train_rgb_modified_img.npy",
    "train_rgb_info.npy",
    "train_ir_modified_img.npy",
    "train_ir_info.npy",
]
OPTIONAL_PREPROCESSED_FILES = ["rgbi2p.pk", "iri2p.pk"]


def read_ids(file_path: Path) -> List[str]:
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    text = file_path.read_text().strip().splitlines()
    if not text:
        return []
    ids = [int(x) for x in text[0].split(",") if x.strip()]
    return [f"{x:04d}" for x in ids]


def find_sysu_source(input_root: Path) -> Path:
    candidates = []
    for p in input_root.rglob("SYSU-MM01"):
        if not p.is_dir():
            continue
        if all((p / cam).exists() for cam in REQUIRED_CAMERAS) and (p / "exp").exists():
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-detect SYSU-MM01 under "
            f"{input_root}. Set --sysu-source explicitly."
        )
    candidates = sorted(candidates, key=lambda x: len(str(x)))
    return candidates[0]


def safe_symlink_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src, target_is_directory=src.is_dir())


def collect_image_paths(root: Path, ids: Iterable[str], cameras: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pid in sorted(ids):
        for cam in cameras:
            img_dir = root / cam / pid
            if img_dir.is_dir():
                paths.extend(sorted([q for q in img_dir.iterdir() if q.is_file()]))
    return paths


def read_samples(paths: List[Path], modal: str, output_root: Path):
    sample_info = []
    img_array = []
    index2path = {}
    desc = f"Processing SYSU {modal} images"
    for index, path in tqdm(list(enumerate(paths)), total=len(paths), desc=desc):
        # Expected path: .../camX/0001/....jpg
        cam_name = path.parent.parent.name
        pid_name = path.parent.name
        camid = int(cam_name.replace("cam", ""))
        pid = int(pid_name)
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
        train_img = np.asarray(img, dtype=np.uint8)
        sample_info.append([index, pid, camid])
        img_array.append(train_img)
        index2path[index] = str(path)
    with (output_root / f"{modal}i2p.pk").open("wb") as f:
        pickle.dump(index2path, f)
    return np.asarray(sample_info, dtype=np.int64), np.asarray(img_array, dtype=np.uint8)


def create_preprocessed(dst: Path) -> None:
    train_ids = read_ids(dst / "exp" / "train_id.txt")
    val_ids = read_ids(dst / "exp" / "val_id.txt")
    train_ids = sorted(set(train_ids + val_ids))
    print(f"SYSU train+val identities: {len(train_ids)}")

    rgb_paths = collect_image_paths(dst, train_ids, RGB_CAMERAS)
    ir_paths = collect_image_paths(dst, train_ids, IR_CAMERAS)
    print(f"SYSU RGB train images: {len(rgb_paths)}")
    print(f"SYSU IR train images: {len(ir_paths)}")

    if not rgb_paths or not ir_paths:
        raise RuntimeError("Could not collect SYSU training images. Check SYSU-MM01 layout.")

    rgb_info, rgb_train = read_samples(rgb_paths, modal="rgb", output_root=dst)
    ir_info, ir_train = read_samples(ir_paths, modal="ir", output_root=dst)

    np.save(dst / "train_rgb_modified_img.npy", rgb_train)
    np.save(dst / "train_rgb_info.npy", rgb_info)
    np.save(dst / "train_ir_modified_img.npy", ir_train)
    np.save(dst / "train_ir_info.npy", ir_info)
    print("Created SYSU preprocessed numpy files.")


def validate(dst: Path) -> None:
    required = [dst / cam for cam in REQUIRED_CAMERAS] + [dst / "exp"]
    required += [dst / name for name in PREPROCESSED_FILES]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing SYSU prepared files:\n" + "\n".join(missing))
    # Light shape check.
    rgb_info = np.load(dst / "train_rgb_info.npy", mmap_mode="r")
    ir_info = np.load(dst / "train_ir_info.npy", mmap_mode="r")
    rgb_img = np.load(dst / "train_rgb_modified_img.npy", mmap_mode="r")
    ir_img = np.load(dst / "train_ir_modified_img.npy", mmap_mode="r")
    print("SYSU prepared at:", dst)
    print("train_rgb_info:", rgb_info.shape, "train_rgb_images:", rgb_img.shape)
    print("train_ir_info:", ir_info.shape, "train_ir_images:", ir_img.shape)
    print("Use training argument: --data-path", dst.parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/kaggle/working/VIREID_Dataset", type=str)
    parser.add_argument("--sysu-source", default="", type=str, help="Path to SYSU-MM01 directory. Empty = auto-detect under --input-root")
    parser.add_argument("--input-root", default="/kaggle/input", type=str)
    parser.add_argument("--force-preprocess", action="store_true", help="Recreate numpy preprocessing even if files exist/link")
    parser.add_argument("--copy-exp", action="store_true", help="Copy exp directory instead of symlink")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    src = Path(args.sysu_source) if args.sysu_source else find_sysu_source(Path(args.input_root))
    if not src.exists():
        raise FileNotFoundError(src)
    dst = data_root / "SYSU-MM01"
    dst.mkdir(parents=True, exist_ok=True)

    print("SYSU source:", src)
    print("SYSU destination:", dst)

    for cam in REQUIRED_CAMERAS:
        safe_symlink_or_copy(src / cam, dst / cam, copy=False)
    safe_symlink_or_copy(src / "exp", dst / "exp", copy=args.copy_exp)

    # Link existing preprocessed files when available and force is not requested.
    if not args.force_preprocess:
        for name in PREPROCESSED_FILES + OPTIONAL_PREPROCESSED_FILES:
            src_file = src / name
            if src_file.exists():
                safe_symlink_or_copy(src_file, dst / name, copy=False)

    if args.force_preprocess or not all((dst / name).exists() for name in PREPROCESSED_FILES):
        create_preprocessed(dst)

    validate(dst)


if __name__ == "__main__":
    main()
