"""Utilities for CRE relation diagnostics.

This module is diagnostic-only. It must not modify CRE outputs, training
losses, relation dictionaries, or model parameters.

Ground-truth identity correspondence, when available from dataset metadata,
is used only for offline evaluation of pseudo-relations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    """Convert tensor/list/array to numpy; return None for missing values."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    try:
        return np.asarray(x)
    except Exception:
        return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        value = float(x)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


def _summary(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": _safe_float(arr.mean()),
        "std": _safe_float(arr.std()),
        "min": _safe_float(arr.min()),
        "max": _safe_float(arr.max()),
    }


def compute_score_stats(scores: Any) -> Dict[str, float]:
    """Compute entropy and top-1/top-2 margin statistics for score matrix.

    Args:
        scores: matrix with shape [num_samples, num_classes]. In WSL, these
            are softmax scores saved by CMA.save(..., mode="scores").
    """
    arr = _to_numpy(scores)
    if arr is None or arr.size == 0 or arr.ndim != 2:
        return {
            "num_samples": 0,
            "num_classes": 0,
            "entropy_mean": 0.0,
            "entropy_std": 0.0,
            "margin_mean": 0.0,
            "margin_std": 0.0,
            "top1_conf_mean": 0.0,
            "top1_conf_std": 0.0,
        }

    arr = arr.astype(np.float64, copy=False)
    eps = 1e-12
    arr = np.clip(arr, eps, None)
    arr = arr / np.clip(arr.sum(axis=1, keepdims=True), eps, None)

    entropy = -(arr * np.log(arr)).sum(axis=1) / np.log(arr.shape[1] + eps)
    top1 = arr.max(axis=1)
    if arr.shape[1] >= 2:
        top2 = np.partition(arr, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
    else:
        margin = np.ones(arr.shape[0], dtype=np.float64)

    return {
        "num_samples": int(arr.shape[0]),
        "num_classes": int(arr.shape[1]),
        "entropy_mean": _safe_float(entropy.mean()),
        "entropy_std": _safe_float(entropy.std()),
        "margin_mean": _safe_float(margin.mean()),
        "margin_std": _safe_float(margin.std()),
        "top1_conf_mean": _safe_float(top1.mean()),
        "top1_conf_std": _safe_float(top1.std()),
    }


def _build_label_to_gt(labels: Any, gts: Any) -> Dict[int, int]:
    labels_np = _to_numpy(labels)
    gts_np = _to_numpy(gts)
    mapping: Dict[int, int] = {}
    if labels_np is None or gts_np is None:
        return mapping
    labels_np = labels_np.reshape(-1)
    gts_np = gts_np.reshape(-1)
    n = min(labels_np.size, gts_np.size)
    for label, gt in zip(labels_np[:n], gts_np[:n]):
        label_i = int(label)
        gt_i = int(gt)
        if label_i not in mapping:
            mapping[label_i] = gt_i
    return mapping


def compute_pair_accuracy(
    pair_dict: Mapping[int, int],
    left_label_to_gt: Mapping[int, int],
    right_label_to_gt: Mapping[int, int],
) -> Dict[str, Any]:
    """Compute offline relation correctness using hidden GT correspondence.

    A pair is correct if both sides map to the same ground-truth identity.
    This metric is for analysis only and must not be used by training.
    """
    total = 0
    known = 0
    correct = 0
    wrong_examples = []
    for left, right in pair_dict.items():
        total += 1
        left_i = int(left)
        right_i = int(right)
        left_gt = left_label_to_gt.get(left_i)
        right_gt = right_label_to_gt.get(right_i)
        if left_gt is None or right_gt is None:
            continue
        known += 1
        is_correct = int(left_gt) == int(right_gt)
        correct += int(is_correct)
        if not is_correct and len(wrong_examples) < 10:
            wrong_examples.append(
                {
                    "left_label": left_i,
                    "right_label": right_i,
                    "left_gt": int(left_gt),
                    "right_gt": int(right_gt),
                }
            )

    return {
        "total": int(total),
        "known": int(known),
        "correct": int(correct),
        "accuracy": _safe_float(correct / known if known > 0 else 0.0),
        "wrong_examples": wrong_examples,
    }


def compute_mutual_match_ratio(
    r2i_pair_dict: Mapping[int, int],
    i2r_pair_dict: Mapping[int, int],
) -> Dict[str, Any]:
    mutual = 0
    for r, i in r2i_pair_dict.items():
        if int(i) in i2r_pair_dict and int(i2r_pair_dict[int(i)]) == int(r):
            mutual += 1
    total = len(r2i_pair_dict)
    return {
        "mutual_count": int(mutual),
        "r2i_total": int(total),
        "i2r_total": int(len(i2r_pair_dict)),
        "ratio": _safe_float(mutual / total if total > 0 else 0.0),
    }


def _prototype_pair_values(cma: Any, pairs: Mapping[int, int]) -> Dict[str, float]:
    if not hasattr(cma, "vis_memory") or not hasattr(cma, "ir_memory"):
        return _summary([])
    if len(pairs) == 0:
        return _summary([])

    with torch.no_grad():
        vis_memory = F.normalize(cma.vis_memory.detach().float().cpu(), dim=1)
        ir_memory = F.normalize(cma.ir_memory.detach().float().cpu(), dim=1)
        sims = []
        num_vis = vis_memory.shape[0]
        num_ir = ir_memory.shape[0]
        for r, i in pairs.items():
            r_i = int(r)
            i_i = int(i)
            if 0 <= r_i < num_vis and 0 <= i_i < num_ir:
                sims.append(float(torch.dot(vis_memory[r_i], ir_memory[i_i]).item()))
    return _summary(sims)


def compute_prototype_similarity_stats(
    cma: Any,
    common_dict: Mapping[int, int],
    specific_dict: Mapping[int, int],
    remain_dict: Mapping[int, int],
    r2i_pair_dict: Mapping[int, int],
    i2r_pair_dict: Mapping[int, int],
) -> Dict[str, Dict[str, float]]:
    i2r_as_r2i = {int(r): int(i) for i, r in i2r_pair_dict.items()}
    return {
        "common": _prototype_pair_values(cma, common_dict),
        "specific": _prototype_pair_values(cma, specific_dict),
        "remain": _prototype_pair_values(cma, remain_dict),
        "r2i_all": _prototype_pair_values(cma, r2i_pair_dict),
        "i2r_all_as_r2i": _prototype_pair_values(cma, i2r_as_r2i),
    }


def _stringify_dict_keys(d: Mapping[int, int]) -> Dict[str, int]:
    return {str(int(k)): int(v) for k, v in d.items()}


def save_relation_diagnostics(
    stats_dir: str,
    epoch: int,
    cma: Any,
    r2i_pair_dict: Mapping[int, int],
    i2r_pair_dict: Mapping[int, int],
    common_dict: Mapping[int, int],
    specific_dict: Mapping[int, int],
    remain_dict: Mapping[int, int],
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """Save one epoch of CRE diagnostics to JSON."""
    Path(stats_dir).mkdir(parents=True, exist_ok=True)

    rgb_label_to_gt = _build_label_to_gt(getattr(cma, "rgb_ids", None), getattr(cma, "rgb_gt", None))
    ir_label_to_gt = _build_label_to_gt(getattr(cma, "ir_ids", None), getattr(cma, "ir_gt", None))

    r2i_acc = compute_pair_accuracy(r2i_pair_dict, rgb_label_to_gt, ir_label_to_gt)
    i2r_acc = compute_pair_accuracy(i2r_pair_dict, ir_label_to_gt, rgb_label_to_gt)
    common_acc = compute_pair_accuracy(common_dict, rgb_label_to_gt, ir_label_to_gt)

    stats: Dict[str, Any] = {
        "epoch": int(epoch),
        "num_r2i_pairs": int(len(r2i_pair_dict)),
        "num_i2r_pairs": int(len(i2r_pair_dict)),
        "num_common": int(len(common_dict)),
        "num_specific": int(len(specific_dict)),
        "num_remain": int(len(remain_dict)),
        "mutual_match": compute_mutual_match_ratio(r2i_pair_dict, i2r_pair_dict),
        "rgb_score_stats": compute_score_stats(getattr(cma, "last_rgb_score", getattr(cma, "vis", None))),
        "ir_score_stats": compute_score_stats(getattr(cma, "last_ir_score", getattr(cma, "ir", None))),
        "r2i_pair_accuracy": r2i_acc,
        "i2r_pair_accuracy": i2r_acc,
        "common_accuracy": common_acc,
        "prototype_similarity": compute_prototype_similarity_stats(
            cma,
            common_dict=common_dict,
            specific_dict=specific_dict,
            remain_dict=remain_dict,
            r2i_pair_dict=r2i_pair_dict,
            i2r_pair_dict=i2r_pair_dict,
        ),
        "pairs_preview": {
            "r2i_first_10": list(_stringify_dict_keys(r2i_pair_dict).items())[:10],
            "i2r_first_10": list(_stringify_dict_keys(i2r_pair_dict).items())[:10],
            "common_first_10": list(_stringify_dict_keys(common_dict).items())[:10],
            "specific_first_10": list(_stringify_dict_keys(specific_dict).items())[:10],
            "remain_first_10": list(_stringify_dict_keys(remain_dict).items())[:10],
        },
    }

    out_path = Path(stats_dir) / f"epoch_{int(epoch):03d}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    msg = (
        f"[relation diagnostics] epoch={epoch} "
        f"common={len(common_dict)} specific={len(specific_dict)} remain={len(remain_dict)} "
        f"mutual={stats['mutual_match']['ratio']:.4f} "
        f"r2i_acc={r2i_acc['accuracy']:.4f} "
        f"i2r_acc={i2r_acc['accuracy']:.4f} "
        f"common_acc={common_acc['accuracy']:.4f}"
    )
    if logger is not None:
        try:
            logger(msg)
        except Exception:
            print(msg)
    else:
        print(msg)

    return stats
