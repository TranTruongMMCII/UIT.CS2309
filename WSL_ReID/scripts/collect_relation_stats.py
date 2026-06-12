#!/usr/bin/env python3
"""Collect CRE relation diagnostic JSON files into a CSV summary.

This version intentionally accepts both --output and --csv-output so that
older Step 3 scripts and newer Step 4 scripts remain compatible.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict


def get_nested(d: Dict[str, Any], path: str, default: Any = "") -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats-dir",
        required=True,
        type=str,
        help="directory containing epoch_*.json files",
    )
    parser.add_argument(
        "--output",
        "--csv-output",
        dest="output",
        default="",
        type=str,
        help="output csv path; default: stats-dir/relation_stats_summary.csv",
    )
    args = parser.parse_args()

    stats_dir = Path(args.stats_dir)
    json_files = sorted(stats_dir.glob("epoch_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No epoch_*.json found in {stats_dir}")

    rows = []
    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as f:
            s = json.load(f)
        rows.append(
            {
                "epoch": get_nested(s, "epoch", ""),
                "num_r2i_pairs": get_nested(s, "num_r2i_pairs", ""),
                "num_i2r_pairs": get_nested(s, "num_i2r_pairs", ""),
                "num_common": get_nested(s, "num_common", ""),
                "num_specific": get_nested(s, "num_specific", ""),
                "num_remain": get_nested(s, "num_remain", ""),
                "mutual_match_ratio": get_nested(s, "mutual_match.ratio", ""),
                "r2i_pair_accuracy": get_nested(s, "r2i_pair_accuracy.accuracy", ""),
                "i2r_pair_accuracy": get_nested(s, "i2r_pair_accuracy.accuracy", ""),
                "common_accuracy": get_nested(s, "common_accuracy.accuracy", ""),
                "rgb_entropy_mean": get_nested(s, "rgb_score_stats.entropy_mean", ""),
                "ir_entropy_mean": get_nested(s, "ir_score_stats.entropy_mean", ""),
                "rgb_margin_mean": get_nested(s, "rgb_score_stats.margin_mean", ""),
                "ir_margin_mean": get_nested(s, "ir_score_stats.margin_mean", ""),
                "proto_common_mean": get_nested(s, "prototype_similarity.common.mean", ""),
                "proto_specific_mean": get_nested(s, "prototype_similarity.specific.mean", ""),
                "proto_remain_mean": get_nested(s, "prototype_similarity.remain.mean", ""),
            }
        )

    output = Path(args.output) if args.output else stats_dir / "relation_stats_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
