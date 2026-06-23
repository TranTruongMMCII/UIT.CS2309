#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/kaggle/working/VIREID_Dataset}
REGDB_SOURCE=${REGDB_SOURCE:-}
RUN_NAME=${RUN_NAME:-upr_v02_filter_regdb}
DEVICE=${DEVICE:-0}
SEED=${SEED:-1}
TRIAL=${TRIAL:-1}
STAGE2_EPOCH=${STAGE2_EPOCH:-15}
MILESTONES=${MILESTONES:-"8 12"}
BATCH_PIDNUM=${BATCH_PIDNUM:-5}
PID_NUMSAMPLE=${PID_NUMSAMPLE:-4}
TEST_BATCH=${TEST_BATCH:-64}
NUM_WORKERS=${NUM_WORKERS:-2}
LR=${LR:-0.00045}
PHASE1_CKPT=${PHASE1_CKPT:?PHASE1_CKPT is required}

UPR_BETA=${UPR_BETA:-0.1}
UPR_GAMMA=${UPR_GAMMA:-0.0}
UPR_MARGIN_WEIGHT=${UPR_MARGIN_WEIGHT:-1.0}
UPR_WARMUP_EPOCH=${UPR_WARMUP_EPOCH:-2}
UPR_FILTER_START_EPOCH=${UPR_FILTER_START_EPOCH:-2}
UPR_FILTER_END_EPOCH=${UPR_FILTER_END_EPOCH:-10}
UPR_FILTER_START_RATIO=${UPR_FILTER_START_RATIO:-0.75}
UPR_FILTER_END_RATIO=${UPR_FILTER_END_RATIO:-1.0}
UPR_FILTER_MIN_PAIRS=${UPR_FILTER_MIN_PAIRS:-40}

cd "$(dirname "$0")/.."

if [[ -n "$REGDB_SOURCE" ]]; then
  python scripts/prepare_regdb_kaggle.py --data-root "$DATA_ROOT" --regdb-source "$REGDB_SOURCE"
else
  python scripts/prepare_regdb_kaggle.py --data-root "$DATA_ROOT"
fi

RELATION_STATS_DIR="../saved_regdb_resnet/${RUN_NAME}_${TRIAL}/relation_stats"

CMD=(python main.py
  --dataset regdb
  --data-path "$DATA_ROOT"
  --debug wsl
  --save-path "$RUN_NAME"
  --arch resnet
  --trial "$TRIAL"
  --seed "$SEED"
  --model-path "$PHASE1_CKPT"
  --stage1-epoch 0
  --stage2-epoch "$STAGE2_EPOCH"
  --lr "$LR"
  --batch-pidnum "$BATCH_PIDNUM"
  --pid-numsample "$PID_NUMSAMPLE"
  --test-batch "$TEST_BATCH"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --milestones $MILESTONES
  --upr-cre
  --upr-beta "$UPR_BETA"
  --upr-gamma "$UPR_GAMMA"
  --upr-margin-weight "$UPR_MARGIN_WEIGHT"
  --upr-warmup-epoch "$UPR_WARMUP_EPOCH"
  --upr-filter
  --upr-filter-start-epoch "$UPR_FILTER_START_EPOCH"
  --upr-filter-end-epoch "$UPR_FILTER_END_EPOCH"
  --upr-filter-start-ratio "$UPR_FILTER_START_RATIO"
  --upr-filter-end-ratio "$UPR_FILTER_END_RATIO"
  --upr-filter-min-pairs "$UPR_FILTER_MIN_PAIRS"
  --save-relation-stats
  --relation-stats-dir "$RELATION_STATS_DIR")

echo "[UPR-CRE v0.2] command:"
printf ' %q' "${CMD[@]}"; echo
"${CMD[@]}"

python scripts/collect_relation_stats.py \
  --stats-dir "$RELATION_STATS_DIR" \
  --csv-output "$RELATION_STATS_DIR/relation_stats_summary.csv"
