#!/usr/bin/env bash
set -euo pipefail

# Diagnostic-only RegDB smoke run. This does not change the CRE algorithm.
# It assumes Step 2a scripts already exist in WSL_ReID/scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DATA_ROOT="${DATA_ROOT:-/kaggle/working/VIREID_Dataset}"
REGDB_SOURCE="${REGDB_SOURCE:-}"
INPUT_ROOT="${INPUT_ROOT:-/kaggle/input}"
RUN_NAME="${RUN_NAME:-relation_diag_regdb_smoke}"
DEVICE="${DEVICE:-0}"

python scripts/prepare_regdb_kaggle.py \
  --source "$REGDB_SOURCE" \
  --output "$DATA_ROOT" \
  --input-root "$INPUT_ROOT"

python scripts/check_kaggle_env.py \
  --data-root "$DATA_ROOT"

python main.py \
  --dataset regdb \
  --data-path "$DATA_ROOT" \
  --debug wsl \
  --save-path "$RUN_NAME" \
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
  --device "$DEVICE" \
  --save-relation-stats \
  --relation-stats-every 1

STATS_DIR="../saved_regdb_resnet/${RUN_NAME}_1/relation_stats"
python scripts/collect_relation_stats.py --stats-dir "$STATS_DIR"

echo "Relation diagnostics written to: $STATS_DIR"
