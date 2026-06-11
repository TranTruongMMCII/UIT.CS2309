#!/usr/bin/env python3
"""Apply minimal Kaggle compatibility patches.

These patches do not implement UPR-CRE. They only fix runtime issues observed on Kaggle:

1. Weak_loss(method='log') mismatch with current Weak_loss signature.
2. Pillow Image.ANTIALIAS removal in newer Pillow versions.
"""
from __future__ import annotations

from pathlib import Path


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> bool:
    if not path.exists():
        return False
    text = path.read_text()
    new_text = text
    for old, new in replacements:
        new_text = new_text.replace(old, new)
    if new_text != text:
        path.write_text(new_text)
        return True
    return False


def main() -> None:
    wsl_dir = Path(__file__).resolve().parents[1]
    changed = []

    p = wsl_dir / "models" / "__init__.py"
    if patch_file(p, [("Weak_loss(method='log')", "Weak_loss()")]):
        changed.append(str(p.relative_to(wsl_dir)))

    antialias_replacement = "(Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)"
    for py in wsl_dir.rglob("*.py"):
        if "scripts/apply_kaggle_compat_patches.py" in str(py):
            continue
        if patch_file(py, [("Image.ANTIALIAS", antialias_replacement)]):
            changed.append(str(py.relative_to(wsl_dir)))

    if changed:
        print("Patched files:")
        for x in changed:
            print(" -", x)
    else:
        print("No compatibility patches were needed.")


if __name__ == "__main__":
    main()
