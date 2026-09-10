#!/usr/bin/env python3
"""Execute ExeBench predictions on the official ten real I/O pairs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from evaluation.exebench.common import (
    DEFAULT_DATASET,
    DEFAULT_METADATA,
    OPTIMIZATIONS,
    evaluate_source,
    read_json,
    save_json,
)


def item_key(row: dict) -> tuple[str, str]:
    return str(row["task_id"]), str(row["type"])


def candidate(row: dict, index: int) -> str:
    candidates = row.get("candidates")
    if candidates and index < len(candidates):
        return candidates[index]
    if index == 0:
        return row.get("output", "")
    return ""


def summarize(rows: list[dict]) -> dict:
    def group_summary(group: list[dict]) -> dict:
        total = len(group)
        tests_total = sum(row["tests_total"] for row in group)
        return {
            "total": total,
            "compile_count": sum(row["compilable"] for row in group),
            "executable_count": sum(row["executable"] for row in group),
            "function_pass_count": sum(row["all_tests_pass"] for row in group),
            "tests_passed": sum(row["tests_passed"] for row in group),
            "tests_total": tests_total,
            "compile_rate": (
                100 * sum(row["compilable"] for row in group) / total
                if total
                else 0.0
            ),
            "executable_rate": (
                100 * sum(row["executable"] for row in group) / total
                if total
                else 0.0
            ),
            "function_pass_rate": (
                100 * sum(row["all_tests_pass"] for row in group) / total
                if total
                else 0.0
            ),
            "io_accuracy": (
                100 * sum(row["tests_passed"] for row in group) / tests_total
                if tests_total
                else 0.0
            ),
            "timeouts": sum(row["timed_out"] for row in group),
        }

    return {
        "overall": group_summary(rows),
        "by_optimization": {
            opt: group_summary([row for row in rows if row["type"] == opt])
            for opt in OPTIMIZATIONS
        },
        "compile_stages": dict(
            sorted(Counter(row["compile_stage"] for row in rows).items())
        ),
    }


def cluster_bootstrap(
    rows: list[dict], samples: int, seed: int
) -> dict[str, list[float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["cluster_id"]].append(row)
    clusters = sorted(grouped)
    rng = np.random.default_rng(seed)
    # Each cluster is summarized once, then integer cluster indices are
    # resampled in bounded batches.  This is equivalent to rebuilding the
    # full row list for every replicate but avoids tens of millions of Python
    # dictionary operations for the 1,941 x 4 ExeBench scope.
    statistics = np.asarray(
        [
            [
                len(grouped[key]),
                sum(row["compilable"] for row in grouped[key]),
                sum(row["all_tests_pass"] for row in grouped[key]),
                sum(row["tests_passed"] for row in grouped[key]),
                sum(row["tests_total"] for row in grouped[key]),
            ]
            for key in clusters
        ],
        dtype=np.int64,
    )
    values = {
        "compile_rate": np.empty(samples, dtype=np.float64),
        "function_pass_rate": np.empty(samples, dtype=np.float64),
        "io_accuracy": np.empty(samples, dtype=np.float64),
    }
    batch_size = 256
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        selected = rng.integers(
            0, len(clusters), size=(stop - start, len(clusters))
        )
        totals = statistics[selected].sum(axis=1)
        values["compile_rate"][start:stop] = 100 * totals[:, 1] / totals[:, 0]
        values["function_pass_rate"][start:stop] = (
            100 * totals[:, 2] / totals[:, 0]
        )
        values["io_accuracy"][start:stop] = (
            100 * totals[:, 3] / totals[:, 4]
        )
    return {
        key: [
            float(np.percentile(observed, 2.5)),
            float(np.percentile(observed, 97.5)),
        ]
        for key, observed in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--objcopy", default="objcopy")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--compile-timeout", type=float, default=30.0)
    parser.add_argument("--test-timeout", type=float, default=5.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = read_json(args.dataset)
    metadata_rows = read_json(args.metadata)
    metadata = {
        int(row["benchmark_index"]): row for row in metadata_rows
    }
    if len(metadata) != len(metadata_rows):
        raise ValueError(f"duplicate benchmark index in {args.metadata}")
    missing_metadata = sorted(
        {
            int(row["benchmark_index"])
            for row in dataset
            if int(row["benchmark_index"]) not in metadata
        }
    )
    if missing_metadata:
        raise ValueError(
            f"{args.metadata} lacks benchmark index {missing_metadata[0]}"
        )
    dataset = [
        {**row, **metadata[int(row["benchmark_index"])]} for row in dataset
    ]
    predictions = read_json(args.predictions)
    expected = [item_key(row) for row in dataset]
    observed = [item_key(row) for row in predictions]
    if len(set(expected)) != len(expected):
        raise ValueError("dataset contains duplicate task/optimization keys")
    if set(expected) != set(observed) or len(observed) != len(expected):
        raise ValueError(
            f"prediction coverage mismatch: expected {len(expected)}, "
            f"observed {len(observed)}, unique observed {len(set(observed))}"
        )
    by_key = {item_key(row): row for row in predictions}
    jobs = [
        (
            row,
            candidate(by_key[item_key(row)], args.candidate_index),
        )
        for row in dataset
    ]

    def work(job: tuple[dict, str]) -> dict:
        row, source = job
        result = evaluate_source(
            row,
            source,
            gcc=args.cc,
            gxx=args.cxx,
            objcopy=args.objcopy,
            compile_timeout=args.compile_timeout,
            test_timeout=args.test_timeout,
        )
        return {
            "task_id": row["task_id"],
            "cluster_id": row["cluster_id"],
            "benchmark_index": row["benchmark_index"],
            "type": row["type"],
            "language": "c",
            "function_name": row["function_name"],
            **result.to_dict(),
        }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        per_case = list(
            tqdm(
                executor.map(work, jobs),
                total=len(jobs),
                desc=args.predictions.stem,
            )
        )
    summary = summarize(per_case)
    bootstrap = cluster_bootstrap(
        per_case, args.bootstrap_samples, args.seed
    )
    payload = {
        "benchmark": "ExeBench v1.01 test_real",
        "language": "C",
        "dataset": str(args.dataset),
        "execution_metadata": str(args.metadata),
        "predictions": str(args.predictions),
        "candidate_policy": (
            f"candidate index {args.candidate_index}; uniform top-1"
            if args.candidate_index == 0
            else f"candidate index {args.candidate_index}"
        ),
        "protocol": {
            "candidate_compile": f"{args.cc} O0",
            "wrapper_compile": f"{args.cxx} O0",
            "objcopy": args.objcopy,
            "oracle": "official real_exe_wrapper and 10 real_io_pairs",
            "function_correct": "all 10 I/O tests match",
            "float_comparison": "math.isclose(rel_tol=1e-9, abs_tol=1e-12)",
            "symbol_alias": "canonical func0 accepted as target function name",
        },
        "summary": summary,
        "cluster_bootstrap_95_ci": bootstrap,
        "per_case": per_case,
    }
    save_json(args.output, payload)
    print(json.dumps(summary, indent=2))
    print(f"Wrote executable metrics to {args.output}")


if __name__ == "__main__":
    main()
