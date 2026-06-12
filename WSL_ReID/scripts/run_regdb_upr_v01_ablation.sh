#!/usr/bin/env bash
set -euo pipefail

# Sequential ablation runner for UPR-CRE v0.1.
# Default is a short smoke ablation. Override STAGE1_EPOCH/STAGE2_EPOCH for dev runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

BASE_RUN_PREFIX="${BASE_RUN_PREFIX:-step4_v01_ablation}"
STAGE1_EPOCH="${STAGE1_EPOCH:-1}"
STAGE2_EPOCH="${STAGE2_EPOCH:-2}"
MILESTONES="${MILESTONES:-1 2}"
SAVE_RELATION_STATS="${SAVE_RELATION_STATS:-1}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-1}"

run_one() {
  local tag="$1"
  local enable="$2"
  local beta="$3"
  local gamma="$4"
  local warmup="$5"

  echo "============================================================"
  echo "[Ablation] ${tag}: enable=${enable}, beta=${beta}, gamma=${gamma}, warmup=${warmup}"
  echo "============================================================"

  RUN_NAME="${BASE_RUN_PREFIX}_${tag}" \
  ENABLE_UPR_CRE="${enable}" \
  UPR_BETA="${beta}" \
  UPR_GAMMA="${gamma}" \
  UPR_WARMUP_EPOCH="${warmup}" \
  STAGE1_EPOCH="${STAGE1_EPOCH}" \
  STAGE2_EPOCH="${STAGE2_EPOCH}" \
  MILESTONES="${MILESTONES}" \
  SAVE_RELATION_STATS="${SAVE_RELATION_STATS}" \
  DEVICE="${DEVICE}" \
  SEED="${SEED}" \
  bash scripts/run_regdb_upr_v01.sh
}

run_one "baseline" "0" "0.0" "0.0" "0"
run_one "proto_only_b02" "1" "0.2" "0.0" "1"
run_one "conf_only_g05" "1" "0.0" "0.5" "1"
run_one "full_b02_g05" "1" "0.2" "0.5" "1"

echo "[Ablation] completed. Relation summaries:"
find ../saved_regdb_resnet -path "*${BASE_RUN_PREFIX}*relation_stats_summary.csv" -print || true
