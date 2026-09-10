#!/usr/bin/env python3
"""Paired cluster-bootstrap comparison of two Decompile-Bench metric files."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


FIELDS = ("compilable", "re_executable", "edit_similarity")


def task_map(payload: dict) -> dict[str, dict]:
    return {row["task_id"]: row for row in payload["per_case"]}


def paired_interval(
    pairs: list[tuple[dict, dict]],
    field: str,
    samples: int,
    seed: int,
) -> tuple[float, list[float]]:
    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for left, right in pairs:
        cluster = str(left.get("cluster_id") or left["task_id"].rsplit("/", 1)[0])
        grouped[cluster].append((left, right))

    def value(row: dict) -> float:
        if field == "edit_similarity":
            return float(row[field])
        return float(bool(row[field]))

    statistics = []
    for cluster in sorted(grouped):
        cluster_pairs = grouped[cluster]
        difference = sum(
            value(left) - value(right) for left, right in cluster_pairs
        )
        statistics.append((difference, len(cluster_pairs)))

    point = 100 * sum(item[0] for item in statistics) / sum(
        item[1] for item in statistics
    )
    randomizer = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = randomizer.choices(statistics, k=len(statistics))
        numerator = sum(item[0] for item in draw)
        denominator = sum(item[1] for item in draw)
        estimates.append(100 * numerator / denominator)
    estimates.sort()
    interval = [
        round(estimates[int(0.025 * (len(estimates) - 1))], 2),
        round(estimates[int(0.975 * (len(estimates) - 1))], 2),
    ]
    return round(point, 2), interval


def summarize_pairs(
    pairs: list[tuple[dict, dict]], samples: int, seed: int
) -> dict:
    differences = {}
    for offset, field in enumerate(FIELDS):
        point, interval = paired_interval(
            pairs, field, samples, seed + offset
        )
        differences[field] = {
            "left_minus_right": point,
            "clustered_ci95": interval,
        }
    left_only = sum(
        bool(left["re_executable"]) and not bool(right["re_executable"])
        for left, right in pairs
    )
    right_only = sum(
        bool(right["re_executable"]) and not bool(left["re_executable"])
        for left, right in pairs
    )
    both_pass = sum(
        bool(left["re_executable"]) and bool(right["re_executable"])
        for left, right in pairs
    )
    return {
        "total": len(pairs),
        "clusters": len(
            {
                str(left.get("cluster_id") or left["task_id"].rsplit("/", 1)[0])
                for left, _ in pairs
            }
        ),
        "differences_percentage_points": differences,
        "run_discordance": {
            "left_only_pass": left_only,
            "right_only_pass": right_only,
            "both_pass": both_pass,
            "neither_pass": len(pairs) - left_only - right_only - both_pass,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    left_payload = json.loads(args.left.read_text(encoding="utf-8"))
    right_payload = json.loads(args.right.read_text(encoding="utf-8"))
    left_rows = task_map(left_payload)
    right_rows = task_map(right_payload)
    common = sorted(left_rows.keys() & right_rows.keys())
    if not common:
        parser.error("metric files have no common task_id")
    if set(left_rows) != set(right_rows):
        parser.error(
            "paired comparison requires identical task sets: "
            f"left-only={len(set(left_rows) - set(right_rows))}, "
            f"right-only={len(set(right_rows) - set(left_rows))}"
        )
    pairs = [(left_rows[task], right_rows[task]) for task in common]
    language_pairs: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for pair in pairs:
        language_pairs[pair[0]["language"]].append(pair)

    payload = {
        "left": args.left_name,
        "right": args.right_name,
        "interpretation": "positive differences favor left",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "overall": summarize_pairs(
            pairs, args.bootstrap_samples, args.seed
        ),
        "by_language": {
            language: summarize_pairs(
                subset, args.bootstrap_samples, args.seed
            )
            for language, subset in sorted(language_pairs.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
