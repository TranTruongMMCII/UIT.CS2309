#!/usr/bin/env bash
set -euo pipefail

# UPR-CRE v0.1 runner for RegDB.
# This script does not modify source code.
# It assumes Step 2a scripts already exist in WSL_ReID/scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REID_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REID_DIR}"

DATA_ROOT="${DATA_ROOT:-/kaggle/working/VIREID_Dataset}"
REGDB_SOURCE="${REGDB_SOURCE:-}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-1}"
TRIAL="${TRIAL:-1}"
RUN_NAME="${RUN_NAME:-upr_v01_regdb_smoke}"
ENABLE_UPR_CRE="${ENABLE_UPR_CRE:-1}"
SAVE_RELATION_STATS="${SAVE_RELATION_STATS:-1}"

STAGE1_EPOCH="${STAGE1_EPOCH:-1}"
STAGE2_EPOCH="${STAGE2_EPOCH:-2}"
MILESTONES="${MILESTONES:-1 2}"
LR="${LR:-0.00045}"
BATCH_PIDNUM="${BATCH_PIDNUM:-2}"
PID_NUMSAMPLE="${PID_NUMSAMPLE:-2}"
TEST_BATCH="${TEST_BATCH:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"

UPR_BETA="${UPR_BETA:-0.2}"
UPR_GAMMA="${UPR_GAMMA:-0.5}"
UPR_MARGIN_WEIGHT="${UPR_MARGIN_WEIGHT:-1.0}"
UPR_WARMUP_EPOCH="${UPR_WARMUP_EPOCH:-1}"

mkdir -p /kaggle/working/run_logs 2>/dev/null || true

echo "[UPR-CRE v0.1] repo: ${REID_DIR}"
echo "[UPR-CRE v0.1] run: ${RUN_NAME}"
echo "[UPR-CRE v0.1] data root: ${DATA_ROOT}"
echo "[UPR-CRE v0.1] RegDB source: ${REGDB_SOURCE:-<auto-detect>}"
echo "[UPR-CRE v0.1] device: ${DEVICE}"
echo "[UPR-CRE v0.1] enable_upr_cre: ${ENABLE_UPR_CRE}"

git rev-parse --short HEAD 2>/dev/null || true
git status --short 2>/dev/null || true

PREPARE_ARGS=(--data-root "${DATA_ROOT}")
if [[ -n "${REGDB_SOURCE}" ]]; then
  PREPARE_ARGS+=(--regdb-source "${REGDB_SOURCE}")
fi
python scripts/prepare_regdb_kaggle.py "${PREPARE_ARGS[@]}"
python scripts/check_kaggle_env.py --data-root "${DATA_ROOT}"

RELATION_STATS_DIR="../saved_regdb_resnet/${RUN_NAME}_${TRIAL}/relation_stats"

CMD=(
  python main.py
  --dataset regdb
  --data-path "${DATA_ROOT}"
  --debug wsl
  --save-path "${RUN_NAME}"
  --arch resnet
  --trial "${TRIAL}"
  --seed "${SEED}"
  --stage1-epoch "${STAGE1_EPOCH}"
  --stage2-epoch "${STAGE2_EPOCH}"
  --lr "${LR}"
  --batch-pidnum "${BATCH_PIDNUM}"
  --pid-numsample "${PID_NUMSAMPLE}"
  --test-batch "${TEST_BATCH}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
)

# MILESTONES is intentionally split by shell word splitting.
# shellcheck disable=SC2206
MILESTONE_ARRAY=(${MILESTONES})
CMD+=(--milestones "${MILESTONE_ARRAY[@]}")

if [[ "${ENABLE_UPR_CRE}" == "1" ]]; then
  CMD+=(
    --upr-cre
    --upr-beta "${UPR_BETA}"
    --upr-gamma "${UPR_GAMMA}"
    --upr-margin-weight "${UPR_MARGIN_WEIGHT}"
    --upr-warmup-epoch "${UPR_WARMUP_EPOCH}"
  )
fi

if [[ "${SAVE_RELATION_STATS}" == "1" ]]; then
  CMD+=(--save-relation-stats --relation-stats-dir "${RELATION_STATS_DIR}")
fi

echo "[UPR-CRE v0.1] command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

if [[ "${SAVE_RELATION_STATS}" == "1" && -d "${RELATION_STATS_DIR}" ]]; then
  python scripts/collect_relation_stats.py \
    --stats-dir "${RELATION_STATS_DIR}" \
    --output "${RELATION_STATS_DIR}/relation_stats_summary.csv"
  echo "[UPR-CRE v0.1] relation summary: ${RELATION_STATS_DIR}/relation_stats_summary.csv"
fi
