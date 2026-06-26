#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/kaggle/working/VIREID_Dataset}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
RUN_NAME=${RUN_NAME:-sysu_baseline_p2s120_paper}
DEVICE=${DEVICE:-0}
SEED=${SEED:-1}
STAGE2_EPOCH=${STAGE2_EPOCH:-120}
LR=${LR:-0.0003}
MILESTONES=${MILESTONES:-"30 70"}
BATCH_PIDNUM=${BATCH_PIDNUM:-8}
PID_NUMSAMPLE=${PID_NUMSAMPLE:-4}
TEST_BATCH=${TEST_BATCH:-128}
NUM_WORKERS=${NUM_WORKERS:-4}
SAVE_RELATION_STATS=${SAVE_RELATION_STATS:-1}
RELATION_STATS_EVERY=${RELATION_STATS_EVERY:-5}

cd "$(dirname "$0")/.."

CMD=(python main.py
  --dataset sysu
  --data-path "$DATA_ROOT"
  --debug wsl
  --save-path "$RUN_NAME"
  --arch resnet
  --seed "$SEED"
  --model-path "$MODEL_PATH"
  --stage1-epoch 0
  --stage2-epoch "$STAGE2_EPOCH"
  --lr "$LR"
  --batch-pidnum "$BATCH_PIDNUM"
  --pid-numsample "$PID_NUMSAMPLE"
  --test-batch "$TEST_BATCH"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --milestones $MILESTONES
  --search-mode all
  --gall-mode single)

if [[ "$SAVE_RELATION_STATS" == "1" ]]; then
  CMD+=(--save-relation-stats --relation-stats-every "$RELATION_STATS_EVERY" \
    --relation-stats-dir "../saved_sysu_resnet/${RUN_NAME}/relation_stats")
fi

echo "${CMD[@]}"
"${CMD[@]}"
