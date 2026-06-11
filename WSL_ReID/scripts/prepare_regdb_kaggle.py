#!/usr/bin/env python3
"""Prepare RegDB Kaggle dataset layout for WSL-VIReID.

The original RegDB loader expects:

    <data-root>/RegDB/idx/train_visible_1.txt
    <data-root>/RegDB/idx/train_thermal_1.txt
    <data-root>/RegDB/Visible/...
    <data-root>/RegDB/Thermal/...

This script creates that layout under /kaggle/working without copying image files.
It uses symlinks to the read-only Kaggle input folder.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def find_regdb_source(input_root: Path = Path("/kaggle/input")) -> Path:
    candidates = []
    for p in input_root.rglob("RegDB"):
        if not p.is_dir():
            continue
        idx_ok = (p / "idx").exists()
        visible_ok = (p / "Visible").exists() or (p / "visible").exists()
        thermal_ok = (p / "Thermal").exists() or (p / "thermal").exists()
        if idx_ok and visible_ok and thermal_ok:
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError("Could not find RegDB under /kaggle/input. Attach the RegDB Kaggle dataset first.")
    candidates = sorted(candidates, key=lambda x: len(str(x)))
    return candidates[0]


def remove_path(p: Path) -> None:
    if p.is_symlink() or p.is_file():
        p.unlink()
    elif p.exists():
        shutil.rmtree(p)


def symlink_or_copytree(src: Path, dst: Path) -> None:
    remove_path(dst)
    try:
        dst.symlink_to(src, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def normalize_index_line(raw: str, expected_modality: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None
    parts = raw.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid index line: {raw!r}")

    rel = parts[0].replace("\\", "/")
    label = parts[1]

    while rel.startswith("./"):
        rel = rel[2:]
    if rel.startswith("RegDB/"):
        rel = rel[len("RegDB/"):]

    chunks = [x for x in rel.split("/") if x]
    if not chunks:
        raise ValueError(f"Invalid relative image path: {rel!r}")

    first = chunks[0].lower()
    if first == "visible":
        chunks[0] = "Visible"
    elif first in {"thermal", "infrared", "ir"}:
        chunks[0] = "Thermal"
    else:
        chunks.insert(0, expected_modality)

    return "/".join(chunks) + " " + label


def prepare_regdb(regdb_source: Path, data_root: Path) -> Path:
    regdb_source = regdb_source.resolve()
    if not regdb_source.exists():
        raise FileNotFoundError(f"RegDB source does not exist: {regdb_source}")

    visible_src = regdb_source / "Visible" if (regdb_source / "Visible").exists() else regdb_source / "visible"
    thermal_src = regdb_source / "Thermal" if (regdb_source / "Thermal").exists() else regdb_source / "thermal"
    idx_src = regdb_source / "idx"

    if not visible_src.exists() or not thermal_src.exists() or not idx_src.exists():
        raise FileNotFoundError(f"RegDB source is missing Visible/Thermal/idx folders: {regdb_source}")

    if data_root.is_symlink() or data_root.is_file():
        data_root.unlink()
    data_root.mkdir(parents=True, exist_ok=True)

    regdb_dst = data_root / "RegDB"
    regdb_dst.mkdir(parents=True, exist_ok=True)

    symlink_or_copytree(visible_src, regdb_dst / "Visible")
    symlink_or_copytree(thermal_src, regdb_dst / "Thermal")
    symlink_or_copytree(visible_src, regdb_dst / "visible")
    symlink_or_copytree(thermal_src, regdb_dst / "thermal")

    idx_dst = regdb_dst / "idx"
    remove_path(idx_dst)
    idx_dst.mkdir(parents=True, exist_ok=True)

    for txt in sorted(idx_src.glob("*.txt")):
        expected = "Thermal" if "thermal" in txt.name.lower() else "Visible"
        lines = []
        for raw in txt.read_text().splitlines():
            line = normalize_index_line(raw, expected)
            if line is not None:
                lines.append(line)
        (idx_dst / txt.name).write_text("\n".join(lines) + "\n")

    required = [
        idx_dst / "train_visible_1.txt",
        idx_dst / "train_thermal_1.txt",
        idx_dst / "test_visible_1.txt",
        idx_dst / "test_thermal_1.txt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required RegDB index files: " + ", ".join(missing))

    for f in required:
        first = f.read_text().splitlines()[0]
        rel = first.split()[0]
        img_path = regdb_dst / rel
        if not img_path.exists():
            raise FileNotFoundError(f"Index file {f.name} points to missing image: {img_path}")

    return regdb_dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regdb-source", default="", help="Path to source RegDB folder. Empty means auto-detect under /kaggle/input.")
    parser.add_argument("--data-root", default="/kaggle/working/VIREID_Dataset", help="Output data root expected by --data-path.")
    args = parser.parse_args()

    regdb_source = Path(args.regdb_source) if args.regdb_source else find_regdb_source()
    data_root = Path(args.data_root)
    regdb_dst = prepare_regdb(regdb_source, data_root)

    print("RegDB source:", regdb_source)
    print("RegDB prepared at:", regdb_dst)
    print("Use training argument: --data-path", data_root)


if __name__ == "__main__":
    main()
