#!/usr/bin/env bash
set -euo pipefail

# Step 8B: soft relation matrix diagnostic only.
# This script does not change the training loss. It enables UPR v0.2 best config
# and logs soft relation distribution statistics.

DATA_ROOT="${DATA_ROOT:-/kaggle/working/VIREID_Dataset}"
REGDB_SOURCE="${REGDB_SOURCE:-}"
PHASE1_CKPT="${PHASE1_CKPT:-}"
RUN_NAME="${RUN_NAME:-upr_v03_softdiag_b01_g00_filter055_p2s5}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-1}"
TRIAL="${TRIAL:-1}"
STAGE2_EPOCH="${STAGE2_EPOCH:-5}"
MILESTONES="${MILESTONES:-8 12}"
LR="${LR:-0.00045}"
BATCH_PIDNUM="${BATCH_PIDNUM:-5}"
PID_NUMSAMPLE="${PID_NUMSAMPLE:-4}"
TEST_BATCH="${TEST_BATCH:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"

UPR_BETA="${UPR_BETA:-0.1}"
UPR_GAMMA="${UPR_GAMMA:-0.0}"
UPR_WARMUP_EPOCH="${UPR_WARMUP_EPOCH:-2}"
UPR_FILTER_START_RATIO="${UPR_FILTER_START_RATIO:-0.55}"
UPR_FILTER_END_RATIO="${UPR_FILTER_END_RATIO:-1.0}"
UPR_FILTER_START_EPOCH="${UPR_FILTER_START_EPOCH:-2}"
UPR_FILTER_END_EPOCH="${UPR_FILTER_END_EPOCH:-10}"
UPR_FILTER_MIN_PAIRS="${UPR_FILTER_MIN_PAIRS:-40}"

UPR_SOFT_TOPK="${UPR_SOFT_TOPK:-3}"
UPR_SOFT_TEMP="${UPR_SOFT_TEMP:-0.5}"
UPR_SOFT_START_EPOCH="${UPR_SOFT_START_EPOCH:-2}"

if [[ -z "${PHASE1_CKPT}" ]]; then
  echo "ERROR: PHASE1_CKPT must point to phase1_model_5.pth"
  exit 1
fi
if [[ ! -f "${PHASE1_CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${PHASE1_CKPT}"
  exit 1
fi

cd "$(dirname "$0")/.."

echo "[Step 8B] repo: $(pwd)"
echo "[Step 8B] run: ${RUN_NAME}"
echo "[Step 8B] checkpoint: ${PHASE1_CKPT}"
git rev-parse --short HEAD || true

PREPARE_ARGS=(--data-root "${DATA_ROOT}")
if [[ -n "${REGDB_SOURCE}" ]]; then
  PREPARE_ARGS+=(--regdb-source "${REGDB_SOURCE}")
fi
python scripts/prepare_regdb_kaggle.py "${PREPARE_ARGS[@]}"
python scripts/check_kaggle_env.py --data-root "${DATA_ROOT}"

RELATION_STATS_DIR="../saved_regdb_resnet/${RUN_NAME}_${TRIAL}/relation_stats"

CMD=(python main.py
  --dataset regdb
  --data-path "${DATA_ROOT}"
  --debug wsl
  --save-path "${RUN_NAME}"
  --arch resnet
  --trial "${TRIAL}"
  --seed "${SEED}"
  --model-path "${PHASE1_CKPT}"
  --stage1-epoch 0
  --stage2-epoch "${STAGE2_EPOCH}"
  --lr "${LR}"
  --batch-pidnum "${BATCH_PIDNUM}"
  --pid-numsample "${PID_NUMSAMPLE}"
  --test-batch "${TEST_BATCH}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --milestones ${MILESTONES}
  --upr-cre
  --upr-beta "${UPR_BETA}"
  --upr-gamma "${UPR_GAMMA}"
  --upr-margin-weight 1.0
  --upr-warmup-epoch "${UPR_WARMUP_EPOCH}"
  --upr-filter
  --upr-filter-start-epoch "${UPR_FILTER_START_EPOCH}"
  --upr-filter-end-epoch "${UPR_FILTER_END_EPOCH}"
  --upr-filter-start-ratio "${UPR_FILTER_START_RATIO}"
  --upr-filter-end-ratio "${UPR_FILTER_END_RATIO}"
  --upr-filter-min-pairs "${UPR_FILTER_MIN_PAIRS}"
  --upr-soft-rel
  --upr-soft-topk "${UPR_SOFT_TOPK}"
  --upr-soft-temp "${UPR_SOFT_TEMP}"
  --upr-soft-start-epoch "${UPR_SOFT_START_EPOCH}"
  --save-relation-stats
  --relation-stats-dir "${RELATION_STATS_DIR}"
)

echo "[Step 8B] command:"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

python scripts/collect_relation_stats.py \
  --stats-dir "${RELATION_STATS_DIR}" \
  --csv-output "${RELATION_STATS_DIR}/relation_stats_summary.csv"

echo "[Step 8B] relation summary: ${RELATION_STATS_DIR}/relation_stats_summary.csv"
cat "${RELATION_STATS_DIR}/relation_stats_summary.csv"
