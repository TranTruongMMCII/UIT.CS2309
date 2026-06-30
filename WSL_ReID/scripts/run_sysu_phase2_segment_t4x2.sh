#!/usr/bin/env bash
set -euo pipefail

# Step 10: SYSU Phase-2 segmented paper-like runner.
# It supports:
#   - first segment: load Phase-1 checkpoint with PHASE1_CKPT_PATH
#   - later segments: load Phase-2 state checkpoints with BASELINE_PHASE2_STATE_PATH / UPR_PHASE2_STATE_PATH
#
# Example first segment:
#   SEGMENT_END_EPOCH=30 bash scripts/run_sysu_phase2_segment_t4x2.sh
# Example next segment:
#   BASELINE_PHASE2_STATE_PATH=/path/phase2_state_30.pth \
#   UPR_PHASE2_STATE_PATH=/path/phase2_state_30.pth \
#   SEGMENT_END_EPOCH=60 bash scripts/run_sysu_phase2_segment_t4x2.sh

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"

DATA_ROOT=${DATA_ROOT:-/kaggle/working/VIREID_Dataset}
PHASE1_CKPT_PATH=${PHASE1_CKPT_PATH:-}
BASELINE_PHASE2_STATE_PATH=${BASELINE_PHASE2_STATE_PATH:-}
UPR_PHASE2_STATE_PATH=${UPR_PHASE2_STATE_PATH:-}

SEGMENT_END_EPOCH=${SEGMENT_END_EPOCH:-30}
SAVE_PHASE2_EVERY=${SAVE_PHASE2_EVERY:-5}
RELATION_STATS_EVERY=${RELATION_STATS_EVERY:-5}
RUN_SUFFIX=${RUN_SUFFIX:-sysu_segment_$(date -u +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)}

LR=${LR:-0.0003}
BATCH_PIDNUM=${BATCH_PIDNUM:-8}
PID_NUMSAMPLE=${PID_NUMSAMPLE:-4}
TEST_BATCH=${TEST_BATCH:-128}
NUM_WORKERS=${NUM_WORKERS:-4}
MILESTONES=${MILESTONES:-"30 70"}
SEED=${SEED:-1}

UPR_BETA=${UPR_BETA:-0.1}
UPR_GAMMA=${UPR_GAMMA:-0.0}
UPR_WARMUP_EPOCH=${UPR_WARMUP_EPOCH:-2}
UPR_FILTER_START_EPOCH=${UPR_FILTER_START_EPOCH:-2}
UPR_FILTER_END_EPOCH=${UPR_FILTER_END_EPOCH:-10}
UPR_FILTER_START_RATIO=${UPR_FILTER_START_RATIO:-0.55}
UPR_FILTER_END_RATIO=${UPR_FILTER_END_RATIO:-1.0}
UPR_FILTER_MIN_PAIRS=${UPR_FILTER_MIN_PAIRS:-40}

RUN_LOG_DIR=${RUN_LOG_DIR:-/kaggle/working/run_logs}
PID_DIR=${PID_DIR:-/kaggle/working/pids}
mkdir -p "$RUN_LOG_DIR" "$PID_DIR"

echo "[SYSU segment] repo: $REPO_DIR"
echo "[SYSU segment] RUN_SUFFIX: $RUN_SUFFIX"
echo "[SYSU segment] SEGMENT_END_EPOCH: $SEGMENT_END_EPOCH"
echo "[SYSU segment] DATA_ROOT: $DATA_ROOT"
echo "[SYSU segment] PHASE1_CKPT_PATH: ${PHASE1_CKPT_PATH:-<empty>}"
echo "[SYSU segment] BASELINE_PHASE2_STATE_PATH: ${BASELINE_PHASE2_STATE_PATH:-<empty>}"
echo "[SYSU segment] UPR_PHASE2_STATE_PATH: ${UPR_PHASE2_STATE_PATH:-<empty>}"

auto_load_args() {
  local state_path="$1"
  if [[ -n "$state_path" ]]; then
    if [[ ! -f "$state_path" ]]; then
      echo "ERROR: phase2 state path not found: $state_path" >&2
      exit 1
    fi
    echo "--phase2-state-path $state_path"
  else
    if [[ -z "$PHASE1_CKPT_PATH" || ! -f "$PHASE1_CKPT_PATH" ]]; then
      echo "ERROR: PHASE1_CKPT_PATH not set or file not found: $PHASE1_CKPT_PATH" >&2
      exit 1
    fi
    echo "--model-path $PHASE1_CKPT_PATH"
  fi
}

BASELINE_SAVE="baseline_sysu_p2s${SEGMENT_END_EPOCH}_${RUN_SUFFIX}"
UPR_SAVE="upr_v02_sysu_p2s${SEGMENT_END_EPOCH}_${RUN_SUFFIX}"

BASELINE_LOAD_ARGS=$(auto_load_args "$BASELINE_PHASE2_STATE_PATH")
UPR_LOAD_ARGS=$(auto_load_args "$UPR_PHASE2_STATE_PATH")

# shellcheck disable=SC2206
MILESTONE_ARGS=($MILESTONES)

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py \
  --dataset sysu \
  --data-path "$DATA_ROOT" \
  --debug wsl \
  --save-path "$BASELINE_SAVE" \
  --arch resnet \
  --seed "$SEED" \
  $BASELINE_LOAD_ARGS \
  --stage1-epoch 0 \
  --stage2-epoch "$SEGMENT_END_EPOCH" \
  --lr "$LR" \
  --batch-pidnum "$BATCH_PIDNUM" \
  --pid-numsample "$PID_NUMSAMPLE" \
  --test-batch "$TEST_BATCH" \
  --num-workers "$NUM_WORKERS" \
  --device 0 \
  --milestones "${MILESTONE_ARGS[@]}" \
  --search-mode all \
  --gall-mode single \
  --save-relation-stats \
  --relation-stats-every "$RELATION_STATS_EVERY" \
  --relation-stats-dir "../saved_sysu_resnet/${BASELINE_SAVE}/relation_stats" \
  --save-phase2-every "$SAVE_PHASE2_EVERY" \
  --phase2-state-dir "../saved_sysu_resnet/${BASELINE_SAVE}/phase2_states" \
  > "${RUN_LOG_DIR}/${BASELINE_SAVE}.log" 2>&1 &

echo $! > "${PID_DIR}/sysu_baseline_segment.pid"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py \
  --dataset sysu \
  --data-path "$DATA_ROOT" \
  --debug wsl \
  --save-path "$UPR_SAVE" \
  --arch resnet \
  --seed "$SEED" \
  $UPR_LOAD_ARGS \
  --stage1-epoch 0 \
  --stage2-epoch "$SEGMENT_END_EPOCH" \
  --lr "$LR" \
  --batch-pidnum "$BATCH_PIDNUM" \
  --pid-numsample "$PID_NUMSAMPLE" \
  --test-batch "$TEST_BATCH" \
  --num-workers "$NUM_WORKERS" \
  --device 0 \
  --milestones "${MILESTONE_ARGS[@]}" \
  --search-mode all \
  --gall-mode single \
  --upr-cre \
  --upr-beta "$UPR_BETA" \
  --upr-gamma "$UPR_GAMMA" \
  --upr-margin-weight 1.0 \
  --upr-warmup-epoch "$UPR_WARMUP_EPOCH" \
  --upr-filter \
  --upr-filter-start-epoch "$UPR_FILTER_START_EPOCH" \
  --upr-filter-end-epoch "$UPR_FILTER_END_EPOCH" \
  --upr-filter-start-ratio "$UPR_FILTER_START_RATIO" \
  --upr-filter-end-ratio "$UPR_FILTER_END_RATIO" \
  --upr-filter-min-pairs "$UPR_FILTER_MIN_PAIRS" \
  --save-relation-stats \
  --relation-stats-every "$RELATION_STATS_EVERY" \
  --relation-stats-dir "../saved_sysu_resnet/${UPR_SAVE}/relation_stats" \
  --save-phase2-every "$SAVE_PHASE2_EVERY" \
  --phase2-state-dir "../saved_sysu_resnet/${UPR_SAVE}/phase2_states" \
  > "${RUN_LOG_DIR}/${UPR_SAVE}.log" 2>&1 &

echo $! > "${PID_DIR}/sysu_upr_segment.pid"

cat > /kaggle/working/sysu_segment_runtime_info.json <<JSON
{
  "run_suffix": "$RUN_SUFFIX",
  "segment_end_epoch": $SEGMENT_END_EPOCH,
  "save_phase2_every": $SAVE_PHASE2_EVERY,
  "relation_stats_every": $RELATION_STATS_EVERY,
  "data_root": "$DATA_ROOT",
  "phase1_ckpt_path": "$PHASE1_CKPT_PATH",
  "baseline_phase2_state_path": "$BASELINE_PHASE2_STATE_PATH",
  "upr_phase2_state_path": "$UPR_PHASE2_STATE_PATH",
  "baseline_save_path": "$BASELINE_SAVE",
  "upr_save_path": "$UPR_SAVE",
  "baseline_log": "${RUN_LOG_DIR}/${BASELINE_SAVE}.log",
  "upr_log": "${RUN_LOG_DIR}/${UPR_SAVE}.log",
  "baseline_pid_file": "${PID_DIR}/sysu_baseline_segment.pid",
  "upr_pid_file": "${PID_DIR}/sysu_upr_segment.pid"
}
JSON

echo "Started SYSU segment jobs:"
echo "baseline PID: $(cat ${PID_DIR}/sysu_baseline_segment.pid)"
echo "UPR PID:      $(cat ${PID_DIR}/sysu_upr_segment.pid)"
echo "runtime info: /kaggle/working/sysu_segment_runtime_info.json"
