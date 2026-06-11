#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${THIS_DIR}/.." && pwd)"
cd "${WSL_DIR}"

DATA_ROOT="${DATA_ROOT:-/kaggle/working/VIREID_Dataset}"
REGDB_SOURCE="${REGDB_SOURCE:-}"
RUN_NAME="${RUN_NAME:-baseline_regdb_s5_s15_step2a}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-1}"
STAGE1_EPOCH="${STAGE1_EPOCH:-5}"
STAGE2_EPOCH="${STAGE2_EPOCH:-15}"
BATCH_PIDNUM="${BATCH_PIDNUM:-5}"
PID_NUMSAMPLE="${PID_NUMSAMPLE:-4}"
TEST_BATCH="${TEST_BATCH:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"

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
  --stage1-epoch "${STAGE1_EPOCH}" \
  --stage2-epoch "${STAGE2_EPOCH}" \
  --milestones 8 12 \
  --lr 0.00045 \
  --batch-pidnum "${BATCH_PIDNUM}" \
  --pid-numsample "${PID_NUMSAMPLE}" \
  --test-batch "${TEST_BATCH}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --seed "${SEED}"
