#!/usr/bin/env python3
"""Evaluate saved decompilation predictions with confidence intervals."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import random
from collections import defaultdict
from pathlib import Path

from common import compile_and_test


_WORKER_CONFIG: dict = {}


def evaluate_one(item: dict) -> dict:
    result = compile_and_test(
        item.get("c_func", ""),
        item.get("c_test", ""),
        item[_WORKER_CONFIG["prediction_field"]],
        prefix=item.get("c_prefix", ""),
        compiler=_WORKER_CONFIG["compiler"],
        timeout=_WORKER_CONFIG["timeout"],
        strict=_WORKER_CONFIG["strict"],
        run_tests=bool(item.get("c_test", "").strip()),
    )
    return {
        "task_id": item.get("task_id", item.get("id")),
        "type": item.get("type", "unknown"),
        "obfuscation": item.get("obfuscation"),
        **result.to_dict(),
    }


def init_worker(config: dict) -> None:
    _WORKER_CONFIG.update(config)


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return [round(100 * max(0.0, center - radius), 2), round(100 * min(1.0, center + radius), 2)]


def bootstrap_macro(
    results: list[dict], field: str, samples: int, seed: int
) -> list[float]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for index, row in enumerate(results):
        task_id = row.get("task_id")
        # HumanEval variants commonly share a task prefix before the final
        # optimization component.  If unavailable, each row is its own unit.
        unit = str(task_id).rsplit("/", 1)[0] if task_id is not None else str(index)
        grouped[unit].append(row)
    units = sorted(grouped)
    if not units:
        return [0.0, 0.0]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [rng.choice(units) for _ in units]
        rows = [row for unit in draw for row in grouped[unit]]
        estimates.append(100 * sum(bool(row[field]) for row in rows) / len(rows))
    estimates.sort()
    lo = estimates[int(0.025 * (len(estimates) - 1))]
    hi = estimates[int(0.975 * (len(estimates) - 1))]
    return [round(lo, 2), round(hi, 2)]


def summarize(rows: list[dict], bootstrap_samples: int, seed: int) -> dict:
    by_opt: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_opt[row["type"]].append(row)

    summary: dict[str, dict] = {}
    for opt, subset in sorted(by_opt.items()):
        total = len(subset)
        compiled = sum(row["compilable"] for row in subset)
        executed = sum(row["re_executable"] for row in subset)
        summary[opt] = {
            "total": total,
            "compile_count": compiled,
            "run_count": executed,
            "compile_rate": round(100 * compiled / total, 2) if total else 0.0,
            "compile_ci95": wilson(compiled, total),
            "run_rate": round(100 * executed / total, 2) if total else 0.0,
            "run_ci95": wilson(executed, total),
        }

    total = len(rows)
    compiled = sum(row["compilable"] for row in rows)
    executed = sum(row["re_executable"] for row in rows)
    summary["mean"] = {
        "total": total,
        "compile_rate": round(100 * compiled / total, 2) if total else 0.0,
        "compile_ci95": bootstrap_macro(rows, "compilable", bootstrap_samples, seed),
        "run_rate": round(100 * executed / total, 2) if total else 0.0,
        "run_ci95": bootstrap_macro(rows, "re_executable", bootstrap_samples, seed + 1),
    }
    return summary


def summarize_by_obfuscation(
    rows: list[dict], bootstrap_samples: int, seed: int
) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("obfuscation") is not None:
            grouped[str(row["obfuscation"])].append(row)
    return {
        name: summarize(subset, bootstrap_samples, seed)
        for name, subset in sorted(grouped.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="gcc")
    parser.add_argument(
        "--prediction-field",
        default="output",
        help="JSON field containing the C candidate (for example initial_output).",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2))
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    items = json.loads(args.predictions.read_text(encoding="utf-8"))
    missing = [
        index
        for index, item in enumerate(items)
        if args.prediction_field not in item
    ]
    if missing:
        parser.error(
            f"--prediction-field {args.prediction_field!r} is missing from "
            f"{len(missing)} items (first index: {missing[0]})"
        )
    config = {
        "compiler": args.compiler,
        "timeout": args.timeout,
        "strict": not args.no_strict,
        "prediction_field": args.prediction_field,
    }
    with multiprocessing.Pool(
        processes=args.workers, initializer=init_worker, initargs=(config,)
    ) as pool:
        per_case = list(pool.imap(evaluate_one, items))

    payload = {
        "protocol": {
            **config,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "summary": summarize(per_case, args.bootstrap_samples, args.seed),
        "by_obfuscation": summarize_by_obfuscation(
            per_case, args.bootstrap_samples, args.seed
        ),
        "per_case": per_case,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
