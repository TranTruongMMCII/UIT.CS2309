#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/kaggle/working/VIREID_Dataset}
SYSU_SOURCE=${SYSU_SOURCE:-}
RUN_NAME=${RUN_NAME:-sysu_phase1_paper}
DEVICE=${DEVICE:-0}
SEED=${SEED:-1}
STAGE1_EPOCH=${STAGE1_EPOCH:-20}
LR=${LR:-0.0003}
MILESTONES=${MILESTONES:-"30 70"}
BATCH_PIDNUM=${BATCH_PIDNUM:-8}
PID_NUMSAMPLE=${PID_NUMSAMPLE:-4}
TEST_BATCH=${TEST_BATCH:-128}
NUM_WORKERS=${NUM_WORKERS:-4}

cd "$(dirname "$0")/.."

PREPARE_ARGS=(--data-root "$DATA_ROOT")
if [[ -n "$SYSU_SOURCE" ]]; then
  PREPARE_ARGS+=(--sysu-source "$SYSU_SOURCE")
fi
python scripts/prepare_sysu_kaggle.py "${PREPARE_ARGS[@]}"
python scripts/check_sysu_env.py --data-root "$DATA_ROOT"

python main.py \
  --dataset sysu \
  --data-path "$DATA_ROOT" \
  --debug wsl \
  --save-path "$RUN_NAME" \
  --arch resnet \
  --seed "$SEED" \
  --stage1-epoch "$STAGE1_EPOCH" \
  --stage2-epoch 0 \
  --lr "$LR" \
  --batch-pidnum "$BATCH_PIDNUM" \
  --pid-numsample "$PID_NUMSAMPLE" \
  --test-batch "$TEST_BATCH" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --milestones $MILESTONES \
  --search-mode all \
  --gall-mode single
