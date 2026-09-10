#!/usr/bin/env python3
"""Evaluate Decompile-Bench predictions with the upstream execution protocol.

Compatibility details intentionally match Decompile-Bench's
``metrics/cal_execute_rate.py``: dependencies, the generated function, and the
provided test are compiled together at ``-O0``; C uses GCC and C++ uses
G++17; the resulting executable must exit successfully. The script adds
per-language summaries and program-clustered confidence intervals without
changing the pass/fail definition.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import multiprocessing
import random
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

try:
    import editdistance
except ImportError as exc:  # pragma: no cover - exercised by CLI environments
    raise SystemExit(
        "evaluate_decompile_bench.py requires editdistance; "
        "install project/requirements.txt"
    ) from exc


_WORKER_CONFIG: dict = {}


def compiler_version(compiler: str) -> str:
    try:
        process = subprocess.run(
            [compiler, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    return process.stdout.splitlines()[0] if process.stdout else "unknown"


def software_versions() -> dict[str, str]:
    versions = {}
    for distribution in (
        "editdistance",
        "torch",
        "transformers",
        "vllm",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not installed"
    return versions


def generation_audit(rows: list[dict], prediction_field: str) -> dict:
    finish_reasons = Counter(
        str(row.get("generation", {}).get("finish_reason"))
        for row in rows
    )
    errors = Counter(
        str(row.get("generation", {}).get("error"))
        for row in rows
        if row.get("generation", {}).get("error")
    )
    context_overflow = sum(
        row.get("generation", {}).get("context_overflow") is True
        or row.get("generation", {}).get("finish_reason")
        == "context_overflow"
        or row.get("generation", {}).get("error") == "context_overflow"
        for row in rows
    )
    return {
        "scored_predictions": len(rows),
        "empty_predictions": sum(not row[prediction_field].strip() for row in rows),
        "context_overflow": context_overflow,
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "generation_errors": dict(sorted(errors.items())),
    }


def normalized_edit_similarity(target: str, prediction: str) -> float:
    """Exact implementation of upstream ``cal_edit_sim.compute_ES``."""

    target_string = "\n".join(
        line.strip() for line in target.splitlines() if line.strip()
    )
    prediction_string = "\n".join(
        line.strip() for line in prediction.splitlines() if line.strip()
    )
    denominator = max(len(target_string), len(prediction_string))
    if denominator == 0:
        return 1.0
    return 1.0 - editdistance.eval(target_string, prediction_string) / denominator


def evaluate_one(item: dict) -> dict:
    prediction = item[_WORKER_CONFIG["prediction_field"]]
    language = item.get("language", "c").lower()
    if language not in {"c", "cpp"}:
        raise ValueError(f"unsupported language: {language!r}")
    compiler = (
        _WORKER_CONFIG["cpp_compiler"]
        if language == "cpp"
        else _WORKER_CONFIG["c_compiler"]
    )
    suffix = ".cpp" if language == "cpp" else ".c"
    source = (
        item.get("c_prefix", "")
        + "\n"
        + prediction
        + "\n"
        + item.get("c_test", "")
    )
    result = {
        "task_id": item.get("task_id"),
        "cluster_id": item.get("cluster_id"),
        "benchmark_index": item.get("benchmark_index"),
        "type": item.get("type", "unknown"),
        "language": language,
        "compilable": False,
        "re_executable": False,
        "compile_returncode": None,
        "run_returncode": None,
        "timed_out": False,
        "stderr": "",
        "edit_similarity": normalized_edit_similarity(
            item.get("c_func", ""), prediction
        ),
    }
    with tempfile.TemporaryDirectory(prefix="dcbench-eval-") as tmp:
        source_path = Path(tmp) / f"candidate{suffix}"
        executable = Path(tmp) / "candidate"
        source_path.write_text(source, encoding="utf-8")
        command = [compiler, _WORKER_CONFIG["optimization"]]
        if language == "c":
            command.extend(_WORKER_CONFIG.get("c_flags", []))
        if language == "cpp":
            command.append("-std=c++17")
        command.extend([str(source_path), "-o", str(executable), "-lm"])
        if language == "cpp":
            command.append(_WORKER_CONFIG["crypto_library"])
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=_WORKER_CONFIG["timeout"],
            )
        except subprocess.TimeoutExpired as exc:
            result["timed_out"] = True
            result["stderr"] = str(exc)
            return result
        result["compile_returncode"] = process.returncode
        result["stderr"] = process.stderr[-4000:]
        if process.returncode != 0:
            return result
        result["compilable"] = True

        try:
            process = subprocess.run(
                [str(executable)],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=_WORKER_CONFIG["timeout"],
                cwd=tmp,
            )
        except subprocess.TimeoutExpired as exc:
            result["timed_out"] = True
            result["stderr"] = (result["stderr"] + "\n" + str(exc))[-4000:]
            return result
        result["run_returncode"] = process.returncode
        result["stderr"] = (result["stderr"] + "\n" + process.stderr)[-4000:]
        result["re_executable"] = process.returncode == 0
    return result


def init_worker(config: dict) -> None:
    _WORKER_CONFIG.update(config)


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if not total:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [
        round(100 * max(0.0, center - radius), 2),
        round(100 * min(1.0, center + radius), 2),
    ]


def program_id(row: dict) -> str:
    if row.get("cluster_id") is not None:
        return str(row["cluster_id"])
    task_id = str(row.get("task_id", ""))
    return task_id.rsplit("/", 1)[0]


def clustered_interval(
    rows: list[dict], field: str, samples: int, seed: int
) -> list[float]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[program_id(row) or str(index)].append(row)
    units = sorted(groups)
    if not units:
        return [0.0, 0.0]
    randomizer = random.Random(seed)
    estimates: list[float] = []
    unit_statistics = []
    for unit in units:
        values = [
            (
                row[field]
                if field == "edit_similarity"
                else float(bool(row[field]))
            )
            for row in groups[unit]
        ]
        unit_statistics.append((sum(values), len(values)))
    for _ in range(samples):
        draw = randomizer.choices(unit_statistics, k=len(unit_statistics))
        numerator = sum(successes for successes, _ in draw)
        denominator = sum(total for _, total in draw)
        estimates.append(100 * numerator / denominator)
    estimates.sort()
    return [
        round(estimates[int(0.025 * (len(estimates) - 1))], 2),
        round(estimates[int(0.975 * (len(estimates) - 1))], 2),
    ]


def point_estimates(rows: list[dict]) -> dict[str, int | float]:
    total = len(rows)
    compiled = sum(row["compilable"] for row in rows)
    executed = sum(row["re_executable"] for row in rows)
    edit_similarity = (
        100 * sum(row["edit_similarity"] for row in rows) / total if total else 0.0
    )
    return {
        "total": total,
        "compile_count": compiled,
        "run_count": executed,
        "compile_rate": round(100 * compiled / total, 2) if total else 0.0,
        "run_rate": round(100 * executed / total, 2) if total else 0.0,
        "edit_similarity": round(edit_similarity, 2),
    }


def summarize_subset(
    rows: list[dict], bootstrap_samples: int, seed: int
) -> dict[str, object]:
    estimates = point_estimates(rows)
    estimates.update(
        {
            "compile_ci95": wilson(
                estimates["compile_count"], estimates["total"]
            ),
            "run_ci95": wilson(estimates["run_count"], estimates["total"]),
            "clustered_ci95": {
                "compile_rate": clustered_interval(
                    rows, "compilable", bootstrap_samples, seed
                ),
                "run_rate": clustered_interval(
                    rows, "re_executable", bootstrap_samples, seed + 1
                ),
                "edit_similarity": clustered_interval(
                    rows, "edit_similarity", bootstrap_samples, seed + 2
                ),
            },
        }
    )
    return estimates


def summarize(rows: list[dict], bootstrap_samples: int, seed: int) -> dict:
    by_optimization: dict[str, list[dict]] = defaultdict(list)
    by_language: dict[str, list[dict]] = defaultdict(list)
    by_language_and_optimization: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_optimization[row["type"]].append(row)
        by_language[row["language"]].append(row)
        by_language_and_optimization[row["language"]][row["type"]].append(row)
    return {
        "overall": summarize_subset(rows, bootstrap_samples, seed),
        "by_optimization": {
            key: summarize_subset(value, bootstrap_samples, seed)
            for key, value in sorted(by_optimization.items())
        },
        "by_language": {
            key: summarize_subset(value, bootstrap_samples, seed)
            for key, value in sorted(by_language.items())
        },
        "by_language_and_optimization": {
            language: {
                optimization: point_estimates(subset)
                for optimization, subset in sorted(optimization_rows.items())
            }
            for language, optimization_rows in sorted(
                by_language_and_optimization.items()
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-field", default="output")
    parser.add_argument(
        "--benchmark-metadata",
        type=Path,
        help=(
            "optional imported benchmark JSON used to attach cluster_id to "
            "prediction files generated before metadata was added"
        ),
    )
    parser.add_argument(
        "--reference-validation",
        type=Path,
        help=(
            "optional metric JSON from evaluating c_func; rows whose reference "
            "does not pass are excluded before scoring model predictions"
        ),
    )
    parser.add_argument("--c-compiler", default="gcc")
    parser.add_argument(
        "--c-flag",
        action="append",
        default=[],
        help="additional C compiler flag; may be passed more than once",
    )
    parser.add_argument("--cpp-compiler", default="g++")
    parser.add_argument(
        "--crypto-library",
        default="-lcrypto",
        help=(
            "OpenSSL link argument from the upstream protocol; systems without "
            "libssl-dev can use '-l:libcrypto.so.3' to select the installed ABI"
        ),
    )
    parser.add_argument("--evaluation-optimization", default="-O0")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2)
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    if args.benchmark_metadata:
        metadata_rows = json.loads(
            args.benchmark_metadata.read_text(encoding="utf-8")
        )
        metadata = {row["task_id"]: row for row in metadata_rows}
        unknown_metadata = [
            row.get("task_id")
            for row in rows
            if row.get("task_id") not in metadata
        ]
        if unknown_metadata:
            parser.error(
                f"--benchmark-metadata lacks {len(unknown_metadata)} tasks "
                f"(first task: {unknown_metadata[0]!r})"
            )
        rows = [
            {
                **row,
                "cluster_id": metadata[row["task_id"]].get("cluster_id"),
            }
            for row in rows
        ]
    excluded_reference_tasks: list[str] = []
    if args.reference_validation:
        validation = json.loads(
            args.reference_validation.read_text(encoding="utf-8")
        )
        reference_pass = {
            row["task_id"]: bool(row["re_executable"])
            for row in validation["per_case"]
        }
        unknown = [
            row.get("task_id")
            for row in rows
            if row.get("task_id") not in reference_pass
        ]
        if unknown:
            parser.error(
                f"--reference-validation lacks {len(unknown)} prediction tasks "
                f"(first task: {unknown[0]!r})"
            )
        excluded_reference_tasks = [
            row["task_id"] for row in rows if not reference_pass[row["task_id"]]
        ]
        rows = [row for row in rows if reference_pass[row["task_id"]]]
    missing = [
        index for index, row in enumerate(rows) if args.prediction_field not in row
    ]
    if missing:
        parser.error(
            f"--prediction-field {args.prediction_field!r} is missing from "
            f"{len(missing)} rows (first index: {missing[0]})"
        )
    config = {
        "prediction_field": args.prediction_field,
        "c_compiler": args.c_compiler,
        "c_flags": args.c_flag,
        "cpp_compiler": args.cpp_compiler,
        "crypto_library": args.crypto_library,
        "optimization": args.evaluation_optimization,
        "timeout": args.timeout,
    }
    with multiprocessing.Pool(
        args.workers, initializer=init_worker, initargs=(config,)
    ) as pool:
        results = list(pool.imap(evaluate_one, rows))

    payload = {
        "protocol": {
            "name": "Decompile-Bench upstream execution protocol",
            **config,
            "c_compiler_version": compiler_version(args.c_compiler),
            "cpp_compiler_version": compiler_version(args.cpp_compiler),
            "software_versions": software_versions(),
            "bootstrap_unit": (
                "underlying MBPP problem (both languages and four "
                "optimization variants)"
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "reference_validation": (
                str(args.reference_validation)
                if args.reference_validation
                else None
            ),
            "excluded_invalid_reference_instances": len(
                excluded_reference_tasks
            ),
            "excluded_invalid_reference_tasks": excluded_reference_tasks,
            "benchmark_metadata": (
                str(args.benchmark_metadata)
                if args.benchmark_metadata
                else None
            ),
        },
        "summary": summarize(results, args.bootstrap_samples, args.seed),
        "generation_audit": generation_audit(rows, args.prediction_field),
        "per_case": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
