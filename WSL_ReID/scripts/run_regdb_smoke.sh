#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${THIS_DIR}/.." && pwd)"
cd "${WSL_DIR}"

DATA_ROOT="${DATA_ROOT:-/kaggle/working/VIREID_Dataset}"
REGDB_SOURCE="${REGDB_SOURCE:-}"
RUN_NAME="${RUN_NAME:-smoke_regdb_step2a}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-1}"

python -m pip install -q -r requirements-kaggle.txt
python scripts/apply_kaggle_compat_patches.py

PREPARE_ARGS=(--data-root "${DATA_ROOT}")
if [[ -n "${REGDB_SOURCE}" ]]; then
  PREPARE_ARGS+=(--regdb-source "${REGDB_SOURCE}")
fi
python scripts/prepare_regdb_kaggle.py "${PREPARE_ARGS[@]}"
python scripts/check_kaggle_env.py --data-root "${DATA_ROOT}"

python main.py \
  --dataset regdb \
  --data-path "${DATA_ROOT}" \
  --debug wsl \
  --save-path "${RUN_NAME}" \
  --arch resnet \
  --trial 1 \
  --stage1-epoch 1 \
  --stage2-epoch 2 \
  --milestones 1 2 \
  --lr 0.00045 \
  --batch-pidnum 2 \
  --pid-numsample 2 \
  --test-batch 32 \
  --num-workers 2 \
  --device "${DEVICE}" \
  --seed "${SEED}"
