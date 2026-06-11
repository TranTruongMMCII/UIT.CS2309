# Step 2a — Private GitHub + reproducible Kaggle RegDB setup

## Purpose

This step makes the Kaggle development environment reproducible for the private fork.
It does not implement UPR-CRE and does not modify the training objective.

## Standard variables

| Variable | Meaning |
|---|---|
| `WORK_DIR` | `/kaggle/working` |
| `REPO_DIR` | `/kaggle/working/WSL-VIReID` |
| `WSL_DIR` | `/kaggle/working/WSL-VIReID/WSL_ReID` |
| `DATA_ROOT` | `/kaggle/working/VIREID_Dataset` |
| `REGDB_SOURCE` | source RegDB folder under `/kaggle/input`, or empty for auto-detect |

## Files added

```text
WSL_ReID/requirements-kaggle.txt
WSL_ReID/scripts/prepare_regdb_kaggle.py
WSL_ReID/scripts/check_kaggle_env.py
WSL_ReID/scripts/apply_kaggle_compat_patches.py
WSL_ReID/scripts/run_regdb_smoke.sh
WSL_ReID/scripts/run_regdb_dev_baseline.sh
docs/experiments/step2a_kaggle_private_env.md
```

## Minimal runtime patches

The compatibility patch script only fixes Kaggle/runtime issues:

1. `Weak_loss(method='log')` -> `Weak_loss()` because the current `Weak_loss` class does not accept `method`.
2. `Image.ANTIALIAS` -> a Pillow-compatible LANCZOS expression.

These are not UPR-CRE method changes.

## Smoke test command

```bash
cd /kaggle/working/WSL-VIReID/WSL_ReID
DATA_ROOT=/kaggle/working/VIREID_Dataset REGDB_SOURCE=/kaggle/input/datasets/xqq027/reg-db/RegDB bash scripts/run_regdb_smoke.sh
```

If `REGDB_SOURCE` is empty, the prepare script auto-detects RegDB under `/kaggle/input`.

## Development baseline command

```bash
cd /kaggle/working/WSL-VIReID/WSL_ReID
DATA_ROOT=/kaggle/working/VIREID_Dataset RUN_NAME=baseline_regdb_s5_s15_step2a bash scripts/run_regdb_dev_baseline.sh
```

The development baseline can take several hours on T4. It is not required every time if a previous local baseline already exists.
