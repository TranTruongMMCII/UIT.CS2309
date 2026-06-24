#!/usr/bin/env python3
"""Collect relation diagnostic JSON files into a flat CSV summary.

This version supports both hard relation diagnostics and Step 8B soft relation
matrix diagnostics. It intentionally accepts both --csv-output and --output,
with --csv-output used as the canonical name in notebooks/scripts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict


def get_nested(d: Dict[str, Any], path: str, default: Any = "") -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def flatten_one(stats: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "epoch": stats.get("epoch", ""),
        "num_r2i_pairs": stats.get("num_r2i_pairs", ""),
        "num_i2r_pairs": stats.get("num_i2r_pairs", ""),
        "num_common": stats.get("num_common", ""),
        "num_specific": stats.get("num_specific", ""),
        "num_remain": stats.get("num_remain", ""),
        "mutual_match_ratio": get_nested(stats, "mutual_match.ratio", ""),
        "r2i_pair_accuracy": get_nested(stats, "r2i_pair_accuracy.accuracy", ""),
        "i2r_pair_accuracy": get_nested(stats, "i2r_pair_accuracy.accuracy", ""),
        "common_accuracy": get_nested(stats, "common_accuracy.accuracy", ""),
        "rgb_entropy_mean": get_nested(stats, "rgb_score_stats.entropy_mean", ""),
        "ir_entropy_mean": get_nested(stats, "ir_score_stats.entropy_mean", ""),
        "rgb_margin_mean": get_nested(stats, "rgb_score_stats.margin_mean", ""),
        "ir_margin_mean": get_nested(stats, "ir_score_stats.margin_mean", ""),
        "proto_common_mean": get_nested(stats, "prototype_similarity.common.mean", ""),
        "proto_specific_mean": get_nested(stats, "prototype_similarity.specific.mean", ""),
        "proto_remain_mean": get_nested(stats, "prototype_similarity.remain.mean", ""),
    }

    # Step 7 filtering stats, if present.
    row.update({
        "filter_rgb_enabled": get_nested(stats, "upr_filter.rgb.enabled", ""),
        "filter_rgb_keep_ratio": get_nested(stats, "upr_filter.rgb.keep_ratio", ""),
        "filter_rgb_original_pairs": get_nested(stats, "upr_filter.rgb.original_pairs", ""),
        "filter_rgb_kept_pairs": get_nested(stats, "upr_filter.rgb.kept_pairs", ""),
        "filter_ir_enabled": get_nested(stats, "upr_filter.ir.enabled", ""),
        "filter_ir_keep_ratio": get_nested(stats, "upr_filter.ir.keep_ratio", ""),
        "filter_ir_original_pairs": get_nested(stats, "upr_filter.ir.original_pairs", ""),
        "filter_ir_kept_pairs": get_nested(stats, "upr_filter.ir.kept_pairs", ""),
    })

    # Step 8B soft relation metrics.
    soft = stats.get("soft_relation", {}) if isinstance(stats.get("soft_relation", {}), dict) else {}
    row.update({
        "soft_enabled": soft.get("enabled", ""),
        "soft_topk": soft.get("topk", ""),
        "soft_temperature": soft.get("temperature", ""),
        "soft_start_epoch": soft.get("start_epoch", ""),
        "soft_r2i_rows": get_nested(stats, "soft_relation.r2i.rows_evaluated", ""),
        "soft_r2i_top1_acc": get_nested(stats, "soft_relation.r2i.top1_accuracy", ""),
        "soft_r2i_topk_acc": get_nested(stats, "soft_relation.r2i.topk_accuracy", ""),
        "soft_r2i_correct_mass_mean": get_nested(stats, "soft_relation.r2i.correct_mass_mean", ""),
        "soft_r2i_entropy_mean": get_nested(stats, "soft_relation.r2i.entropy_mean", ""),
        "soft_i2r_rows": get_nested(stats, "soft_relation.i2r.rows_evaluated", ""),
        "soft_i2r_top1_acc": get_nested(stats, "soft_relation.i2r.top1_accuracy", ""),
        "soft_i2r_topk_acc": get_nested(stats, "soft_relation.i2r.topk_accuracy", ""),
        "soft_i2r_correct_mass_mean": get_nested(stats, "soft_relation.i2r.correct_mass_mean", ""),
        "soft_i2r_entropy_mean": get_nested(stats, "soft_relation.i2r.entropy_mean", ""),
    })

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-dir", required=True, type=str)
    parser.add_argument(
        "--csv-output", "--output",
        dest="csv_output",
        default="",
        type=str,
        help="output csv path; default: stats-dir/relation_stats_summary.csv",
    )
    args = parser.parse_args()

    stats_dir = Path(args.stats_dir)
    if not stats_dir.exists():
        raise FileNotFoundError(stats_dir)

    files = sorted(stats_dir.glob("epoch_*.json"))
    if not files:
        raise FileNotFoundError(f"No epoch_*.json files in {stats_dir}")

    rows = []
    for f in files:
        with f.open("r", encoding="utf-8") as fp:
            rows.append(flatten_one(json.load(fp)))

    out = Path(args.csv_output) if args.csv_output else stats_dir / "relation_stats_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Stable column order: use keys from first row, then include any later extras.
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with out.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(out)


if __name__ == "__main__":
    main()
