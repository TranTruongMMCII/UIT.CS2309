"""Soft relation diagnostic utilities for UPR-CRE Step 8B.

This module is diagnostic-only. It reads soft relation matrices stored in `CMA`
and computes offline statistics using ground-truth correspondence mappings.
Ground truth is used only for evaluation/logging and never for training.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence
import math
import numpy as np


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _normalize_distribution(row: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row = np.asarray(row, dtype=np.float64)
    row = np.clip(row, 0.0, None)
    s = float(row.sum())
    if s <= eps:
        return np.zeros_like(row, dtype=np.float64)
    return row / s


def _entropy(row: np.ndarray, eps: float = 1e-12) -> float:
    p = _normalize_distribution(row, eps=eps)
    nz = p[p > eps]
    if nz.size <= 1:
        return 0.0
    return float(-(nz * np.log(nz)).sum() / np.log(float(nz.size)))


def _directional_soft_stats(
    matrix: Optional[np.ndarray],
    valid_rows: Optional[Sequence[bool]],
    src_label_to_gt: Mapping[int, int],
    tgt_label_to_gt: Mapping[int, int],
    topk: int,
) -> Dict[str, Any]:
    if matrix is None:
        return {
            "enabled": False,
            "reason": "matrix_missing",
            "rows_evaluated": 0,
            "top1_accuracy": 0.0,
            "topk_accuracy": 0.0,
            "correct_mass_mean": 0.0,
            "entropy_mean": 0.0,
        }

    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] == 0 or mat.shape[1] == 0:
        return {
            "enabled": False,
            "reason": "invalid_matrix_shape",
            "shape": list(mat.shape),
            "rows_evaluated": 0,
            "top1_accuracy": 0.0,
            "topk_accuracy": 0.0,
            "correct_mass_mean": 0.0,
            "entropy_mean": 0.0,
        }

    if valid_rows is None:
        valid = mat.sum(axis=1) > 0
    else:
        valid = np.asarray(valid_rows, dtype=bool)
        if valid.shape[0] != mat.shape[0]:
            valid = mat.sum(axis=1) > 0

    topk = int(max(1, min(topk, mat.shape[1])))

    top1_correct = []
    topk_correct = []
    correct_mass = []
    entropies = []
    rows_evaluated = 0
    rows_with_correct_target = 0

    # Pre-index target labels by GT identity.
    gt_to_targets: Dict[int, list[int]] = {}
    for tgt_label, gt in tgt_label_to_gt.items():
        try:
            tgt_i = int(tgt_label)
            gt_i = int(gt)
        except Exception:
            continue
        if 0 <= tgt_i < mat.shape[1]:
            gt_to_targets.setdefault(gt_i, []).append(tgt_i)

    for src_label in range(mat.shape[0]):
        if not bool(valid[src_label]):
            continue
        if src_label not in src_label_to_gt:
            continue
        row = _normalize_distribution(mat[src_label])
        if float(row.sum()) <= 0:
            continue

        rows_evaluated += 1
        src_gt = int(src_label_to_gt[src_label])
        correct_targets = gt_to_targets.get(src_gt, [])
        if correct_targets:
            rows_with_correct_target += 1

        top1 = int(np.argmax(row))
        if topk >= row.shape[0]:
            topk_idx = np.argsort(row)[::-1]
        else:
            # argpartition is faster, then sort only the selected top-k.
            idx = np.argpartition(row, -topk)[-topk:]
            topk_idx = idx[np.argsort(row[idx])[::-1]]
        topk_set = set(int(x) for x in topk_idx[:topk])
        correct_set = set(int(x) for x in correct_targets)

        top1_correct.append(1.0 if top1 in correct_set else 0.0)
        topk_correct.append(1.0 if topk_set.intersection(correct_set) else 0.0)
        correct_mass.append(float(row[correct_targets].sum()) if correct_targets else 0.0)
        entropies.append(_entropy(row))

    return {
        "enabled": True,
        "shape": list(mat.shape),
        "topk": int(topk),
        "rows_evaluated": int(rows_evaluated),
        "rows_with_correct_target": int(rows_with_correct_target),
        "coverage": float(rows_evaluated / max(1, mat.shape[0])),
        "top1_accuracy": float(np.mean(top1_correct)) if top1_correct else 0.0,
        "topk_accuracy": float(np.mean(topk_correct)) if topk_correct else 0.0,
        "correct_mass_mean": float(np.mean(correct_mass)) if correct_mass else 0.0,
        "correct_mass_median": float(np.median(correct_mass)) if correct_mass else 0.0,
        "entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "entropy_median": float(np.median(entropies)) if entropies else 0.0,
    }


def compute_soft_relation_diagnostics(
    cma: Any,
    rgb_label_to_gt: Mapping[int, int],
    ir_label_to_gt: Mapping[int, int],
) -> Dict[str, Any]:
    """Compute offline metrics for soft cross-modal relation matrices.

    Expected optional fields in `cma`:
      - soft_r2i_matrix: [num_rgb_ids, num_ir_ids]
      - soft_i2r_matrix: [num_ir_ids, num_rgb_ids]
      - soft_r2i_valid_rows / soft_i2r_valid_rows
      - upr_soft_topk

    Ground truth mapping is only used for diagnostic metrics.
    """
    topk = int(getattr(cma, "upr_soft_topk", 3))

    r2i = _directional_soft_stats(
        matrix=getattr(cma, "soft_r2i_matrix", None),
        valid_rows=getattr(cma, "soft_r2i_valid_rows", None),
        src_label_to_gt=rgb_label_to_gt,
        tgt_label_to_gt=ir_label_to_gt,
        topk=topk,
    )

    i2r = _directional_soft_stats(
        matrix=getattr(cma, "soft_i2r_matrix", None),
        valid_rows=getattr(cma, "soft_i2r_valid_rows", None),
        src_label_to_gt=ir_label_to_gt,
        tgt_label_to_gt=rgb_label_to_gt,
        topk=topk,
    )

    return {
        "enabled": bool(getattr(cma, "upr_soft_rel", False)),
        "topk": int(topk),
        "temperature": _safe_float(getattr(cma, "upr_soft_temp", None), 0.0),
        "start_epoch": int(getattr(cma, "upr_soft_start_epoch", 0)),
        "r2i": r2i,
        "i2r": i2r,
        "last_soft_stats": getattr(cma, "last_soft_relation_stats", {}),
    }
