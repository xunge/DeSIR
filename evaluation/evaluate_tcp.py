#!/usr/bin/env python3
"""Measure macro-averaged Test Case Pass Rate for saved C predictions.

HumanEval-Decompile and MBPP-Decompile package several checks in one C
``main``.  Their legacy RE metric only records whether that complete process
exits successfully.  This evaluator rewrites top-level checks to record
success without aborting on a failed assertion.  Checks run in their original
program order because many converted harnesses put candidate calls and
in-place mutations immediately before an assertion.  TCP is computed per
function before averaging:

    TCP = mean_f(passed_tests_f / total_tests_f)

If a process crashes or times out, checks completed before the failure remain
credited and all unobserved checks fail.  Functions whose harness contains no
checkable oracle are excluded rather than assigning a value to 0/0.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from evaluation.common import extract_c_code, split_includes


OPTIMIZATIONS = ("O0", "O1", "O2", "O3")
MARKER_RE = re.compile(r"__DECIR_TCP__\s+(\d+)\s+(\d+)")
_WORKER_CONFIG: dict = {}


@dataclass
class RunResult:
    compilable: bool = False
    run_returncode: int | None = None
    timed_out: bool = False
    tests_passed: int = 0
    tests_observed: int = 0
    stderr: str = ""


def code_mask(source: str) -> str:
    """Return a same-length view with comments and literals blanked."""

    result = list(source)
    index = 0
    state = "code"
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char == "/" and following == "/":
                result[index] = result[index + 1] = " "
                index += 2
                state = "line_comment"
                continue
            if char == "/" and following == "*":
                result[index] = result[index + 1] = " "
                index += 2
                state = "block_comment"
                continue
            if char == '"':
                result[index] = " "
                state = "string"
            elif char == "'":
                result[index] = " "
                state = "character"
        elif state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                result[index] = " "
        elif state == "block_comment":
            if char == "*" and following == "/":
                result[index] = result[index + 1] = " "
                index += 2
                state = "code"
                continue
            if char != "\n":
                result[index] = " "
        elif state in {"string", "character"}:
            if char == "\\" and following:
                result[index] = result[index + 1] = " "
                index += 2
                continue
            terminator = '"' if state == "string" else "'"
            if char == terminator:
                result[index] = " "
                state = "code"
            elif char != "\n":
                result[index] = " "
        index += 1
    return "".join(result)


def closing_delimiter(masked: str, opening: int, left: str, right: str) -> int:
    depth = 0
    for index in range(opening, len(masked)):
        if masked[index] == left:
            depth += 1
        elif masked[index] == right:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unbalanced {left}{right} delimiter at byte {opening}")


def main_body(source: str, masked: str) -> tuple[int, int]:
    match = re.search(r"\bmain\s*\(", masked)
    if not match:
        raise ValueError("test harness has no main function")
    opening_parenthesis = masked.find("(", match.start())
    closing_parenthesis = closing_delimiter(
        masked, opening_parenthesis, "(", ")"
    )
    opening_brace = masked.find("{", closing_parenthesis)
    if opening_brace < 0:
        raise ValueError("main declaration has no function body")
    closing_brace = closing_delimiter(masked, opening_brace, "{", "}")
    return opening_brace + 1, closing_brace


def calls_in_range(
    source: str,
    masked: str,
    name: str,
    start: int,
    stop: int,
) -> list[tuple[int, int, str]]:
    pattern = re.compile(rf"\b{re.escape(name)}\s*\(")
    calls = []
    for match in pattern.finditer(masked, start, stop):
        opening = masked.find("(", match.start(), stop)
        closing = closing_delimiter(masked, opening, "(", ")")
        if closing >= stop:
            continue
        calls.append((match.start(), closing + 1, source[opening + 1 : closing]))
    return calls


def instrument_tests(source: str) -> tuple[str, str, int]:
    """Rewrite checkable calls in ``main`` and return kind/static site count."""

    masked = code_mask(source)
    start, stop = main_body(source, masked)
    standard = calls_in_range(source, masked, "assert", start, stop)
    replacements: list[tuple[int, int, str]]
    kind: str
    if standard:
        replacements = [
            (left, right, f"TCP_CHECK_EXPR(({arguments}))")
            for left, right, arguments in standard
        ]
        kind = "assert"
    else:
        array_checks = calls_in_range(
            source, masked, "assertArrayEquals", start, stop
        )
        if array_checks:
            replacements = [
                (
                    left,
                    right,
                    f"TCP_CHECK_CALL(assertArrayEquals({arguments}))",
                )
                for left, right, arguments in array_checks
            ]
            kind = "assertArrayEquals"
        else:
            # HumanEval task 74 has three boolean comparator calls whose
            # return values were accidentally ignored by the converted
            # harness.  They are nevertheless explicit, checkable cases.
            same_checks = calls_in_range(source, masked, "issame", start, stop)
            if same_checks:
                replacements = [
                    (
                        left,
                        right,
                        f"TCP_CHECK_EXPR((issame({arguments})))",
                    )
                    for left, right, arguments in same_checks
                ]
                kind = "issame"
            else:
                return source, "no_checkable_oracle", 0
    rewritten = source
    for left, right, replacement in reversed(replacements):
        rewritten = rewritten[:left] + replacement + rewritten[right:]
    return rewritten, kind, len(replacements)


def instrument_helper_asserts(source: str) -> str:
    """Make assertions inside an array-comparison helper non-aborting."""

    masked = code_mask(source)
    calls = calls_in_range(source, masked, "assert", 0, len(source))
    rewritten = source
    for left, right, arguments in reversed(calls):
        rewritten = (
            rewritten[:left]
            + f"TCP_HELPER_ASSERT(({arguments}))"
            + rewritten[right:]
        )
    return rewritten


def instrumentation() -> str:
    return """
#include <stdio.h>
#include <stdlib.h>

static unsigned long long tcp_passed = 0;
static unsigned long long tcp_total = 0;
static int tcp_helper_failed = 0;

static void tcp_report(void) {
    fprintf(stderr, "__DECIR_TCP__ %llu %llu\\n", tcp_passed, tcp_total);
    fflush(stderr);
}

__attribute__((constructor))
static void tcp_initialize(void) {
    atexit(tcp_report);
}

#define TCP_CHECK_EXPR(...) do {                                       \\
    ++tcp_total;                                                       \\
    tcp_passed += !!(__VA_ARGS__);                                    \\
    tcp_report();                                                     \\
} while (0)

#define TCP_CHECK_CALL(...) do {                                       \\
    tcp_helper_failed = 0;                                            \\
    __VA_ARGS__;                                                      \\
    ++tcp_total;                                                       \\
    tcp_passed += !tcp_helper_failed;                                 \\
    tcp_report();                                                     \\
} while (0)

#define TCP_HELPER_ASSERT(...) do {                                    \\
    if (!(__VA_ARGS__))                                               \\
        tcp_helper_failed = 1;                                        \\
} while (0)
"""


def run_source(
    item: dict,
    prediction: str,
    *,
    compiler: str,
    compile_timeout: float,
    run_timeout: float,
) -> tuple[RunResult, str, int]:
    tests, kind, static_sites = instrument_tests(str(item.get("c_test", "")))
    if not static_sites:
        return RunResult(stderr="no checkable test oracle"), kind, 0
    prediction = extract_c_code(prediction)
    if not prediction:
        return RunResult(stderr="prediction contains no C source"), kind, static_sites
    includes, (_, tests, prediction, prefix) = split_includes(
        str(item.get("c_func", "")),
        tests,
        prediction,
        str(item.get("c_prefix", "")),
    )
    if kind == "assertArrayEquals":
        prefix = instrument_helper_asserts(prefix)
    combined_source = (
        includes
        + "\n"
        + instrumentation()
        + "\n"
        + prefix
        + "\n"
        + prediction
        + "\n"
        + tests
        + "\n"
    )
    result = RunResult()
    with tempfile.TemporaryDirectory(prefix="decir-tcp-") as temporary:
        root = Path(temporary)
        source_path = root / "candidate.c"
        executable = root / "candidate"
        source_path.write_text(combined_source, encoding="utf-8")
        try:
            process = subprocess.run(
                [
                    compiler,
                    "-O0",
                    str(source_path),
                    "-o",
                    str(executable),
                    "-lm",
                ],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=compile_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            result.timed_out = True
            result.stderr = str(exc)
            return result, kind, static_sites
        result.stderr = process.stderr[-4000:]
        if process.returncode != 0:
            return result, kind, static_sites
        result.compilable = True
        try:
            process = subprocess.run(
                [str(executable)],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=run_timeout,
                cwd=root,
                start_new_session=True,
            )
            result.run_returncode = process.returncode
            result.stderr = (result.stderr + "\n" + process.stderr)[-4000:]
        except subprocess.TimeoutExpired as exc:
            result.timed_out = True
            partial = exc.stderr or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            result.stderr = (
                result.stderr + "\n" + partial + "\n" + str(exc)
            )[-4000:]
    markers = MARKER_RE.findall(result.stderr)
    if markers:
        passed, observed = markers[-1]
        result.tests_passed = int(passed)
        result.tests_observed = int(observed)
    return result, kind, static_sites


def prediction(row: dict, field: str, index: int) -> str:
    value = row.get(field, "")
    if isinstance(value, list):
        return str(value[index]) if index < len(value) else ""
    return str(value) if index == 0 else ""


def reference_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("c_prefix", "")),
        str(row.get("c_func", "")),
        str(row.get("c_test", "")),
    )


def init_worker(config: dict) -> None:
    _WORKER_CONFIG.update(config)


def run_reference(row: dict) -> tuple[tuple[str, str, str], dict]:
    result, kind, sites = run_source(
        row,
        str(row.get("c_func", "")),
        compiler=_WORKER_CONFIG["compiler"],
        compile_timeout=_WORKER_CONFIG["compile_timeout"],
        run_timeout=_WORKER_CONFIG["run_timeout"],
    )
    valid = (
        sites > 0
        and result.compilable
        and not result.timed_out
        and result.run_returncode == 0
        and result.tests_observed > 0
        and result.tests_passed == result.tests_observed
    )
    return reference_key(row), {
        "valid": valid,
        "kind": kind,
        "static_sites": sites,
        "tests_total": result.tests_observed,
        "result": asdict(result),
    }


def run_candidate(job: tuple[dict, dict]) -> dict:
    row, reference = job
    result, kind, sites = run_source(
        row,
        prediction(
            row,
            _WORKER_CONFIG["prediction_field"],
            _WORKER_CONFIG["candidate_index"],
        ),
        compiler=_WORKER_CONFIG["compiler"],
        compile_timeout=_WORKER_CONFIG["compile_timeout"],
        run_timeout=_WORKER_CONFIG["run_timeout"],
    )
    expected = int(reference["tests_total"])
    passed = min(result.tests_passed, expected)
    return {
        "task_id": row.get("task_id"),
        "cluster_id": row.get("cluster_id"),
        "benchmark_index": row.get("benchmark_index"),
        "type": str(row.get("type", "unknown")),
        "obfuscation": row.get("obfuscation"),
        "language": str(row.get("language", "c")).lower(),
        "compilable": result.compilable,
        "run_returncode": result.run_returncode,
        "timed_out": result.timed_out,
        "test_kind": kind,
        "static_test_sites": sites,
        "tests_passed": passed,
        "tests_total": expected,
        "tests_observed": result.tests_observed,
        "test_count_matches_reference": result.tests_observed == expected,
        "all_tests_pass": (
            result.run_returncode == 0
            and not result.timed_out
            and result.tests_observed == expected
            and passed == expected
        ),
        "tcp": passed / expected,
        "stderr": result.stderr,
    }


def subset_summary(rows: list[dict]) -> dict:
    total = len(rows)
    tcp = 100 * sum(row["tcp"] for row in rows) / total if total else 0.0
    return {
        "functions": total,
        "tcp": round(tcp, 2),
        "all_tests_pass_count": sum(row["all_tests_pass"] for row in rows),
        "all_tests_pass_rate": (
            round(
                100 * sum(row["all_tests_pass"] for row in rows) / total,
                2,
            )
            if total
            else 0.0
        ),
        "compiled_count": sum(row["compilable"] for row in rows),
        "tests_passed": sum(row["tests_passed"] for row in rows),
        "tests_total": sum(row["tests_total"] for row in rows),
        "test_count_mismatches": sum(
            not row["test_count_matches_reference"] for row in rows
        ),
        "timeouts": sum(row["timed_out"] for row in rows),
    }


def summarize(rows: list[dict]) -> dict:
    present_optimizations = [
        optimization
        for optimization in OPTIMIZATIONS
        if any(row["type"] == optimization for row in rows)
    ]
    by_optimization = {
        optimization: subset_summary(
            [row for row in rows if row["type"] == optimization]
        )
        for optimization in present_optimizations
    }
    summary = {
        "overall": subset_summary(rows),
        "by_optimization": by_optimization,
        "macro_average_of_optimization_tcp": round(
            sum(
                by_optimization[optimization]["tcp"]
                for optimization in present_optimizations
            )
            / len(present_optimizations),
            2,
        ),
    }
    obfuscations = sorted(
        {
            str(row["obfuscation"])
            for row in rows
            if row.get("obfuscation") is not None
        }
    )
    if obfuscations:
        summary["by_obfuscation"] = {
            obfuscation: subset_summary(
                [
                    row
                    for row in rows
                    if str(row.get("obfuscation")) == obfuscation
                ]
            )
            for obfuscation in obfuscations
        }
        summary["by_obfuscation_and_optimization"] = {
            obfuscation: {
                optimization: subset_summary(
                    [
                        row
                        for row in rows
                        if str(row.get("obfuscation")) == obfuscation
                        and row["type"] == optimization
                    ]
                )
                for optimization in present_optimizations
            }
            for obfuscation in obfuscations
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("humaneval", "mbpp"), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input-compiler", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-field", default="output")
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument(
        "--allow-partial-optimizations",
        action="store_true",
        help=(
            "Accept a balanced subset of O0--O3. This is required for "
            "released systems such as SLaDe that provide only O0/O3 "
            "checkpoints."
        ),
    )
    parser.add_argument("--compiler", default="gcc")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(32, multiprocessing.cpu_count() // 2)),
    )
    parser.add_argument("--compile-timeout", type=float, default=30.0)
    parser.add_argument("--run-timeout", type=float, default=10.0)
    args = parser.parse_args()

    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        parser.error("--predictions must contain a JSON list")
    if args.benchmark == "mbpp":
        rows = [
            row
            for row in rows
            if str(row.get("language", "c")).lower() == "c"
        ]
    counts = Counter(str(row.get("type")) for row in rows)
    valid_optimization_scope = (
        bool(counts)
        and set(counts).issubset(OPTIMIZATIONS)
        and len(set(counts.values())) == 1
        and (
            args.allow_partial_optimizations
            or set(counts) == set(OPTIMIZATIONS)
        )
    )
    if not valid_optimization_scope:
        parser.error(f"unbalanced O0--O3 prediction scope: {dict(counts)}")
    missing = [
        index
        for index, row in enumerate(rows)
        if args.prediction_field not in row
    ]
    if missing:
        parser.error(
            f"field {args.prediction_field!r} missing at row {missing[0]}"
        )

    config = {
        "compiler": args.compiler,
        "compile_timeout": args.compile_timeout,
        "run_timeout": args.run_timeout,
        "prediction_field": args.prediction_field,
        "candidate_index": args.candidate_index,
    }
    unique_references: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        unique_references.setdefault(reference_key(row), row)
    with multiprocessing.Pool(
        args.workers, initializer=init_worker, initargs=(config,)
    ) as pool:
        evaluated_references = dict(
            pool.imap(run_reference, unique_references.values())
        )
    invalid_keys = {
        key for key, result in evaluated_references.items() if not result["valid"]
    }
    valid_rows = [
        row for row in rows if reference_key(row) not in invalid_keys
    ]
    jobs = [
        (row, evaluated_references[reference_key(row)])
        for row in valid_rows
    ]
    with multiprocessing.Pool(
        args.workers, initializer=init_worker, initargs=(config,)
    ) as pool:
        per_case = list(pool.imap(run_candidate, jobs))

    invalid_rows = [
        {
            "task_id": row.get("task_id"),
            "cluster_id": row.get("cluster_id"),
            "type": row.get("type"),
            "obfuscation": row.get("obfuscation"),
            **evaluated_references[reference_key(row)],
        }
        for row in rows
        if reference_key(row) in invalid_keys
    ]
    payload = {
        "benchmark": (
            "HumanEval-Decompile"
            if args.benchmark == "humaneval"
            else "MBPP-Decompile"
        ),
        "language": "C",
        "model": args.model_name,
        "input_compiler": args.input_compiler,
        "predictions": str(args.predictions),
        "protocol": {
            "definition": (
                "mean over functions of passed test cases / total test cases"
            ),
            "aggregation": "function macro-average",
            "execution_compiler": args.compiler,
            "candidate_index": args.candidate_index,
            "prediction_field": args.prediction_field,
            "optimization_scope": sorted(
                counts, key=OPTIMIZATIONS.index
            ),
            "test_execution": (
                "original harness order; failed assertions record zero "
                "without aborting; unobserved checks after crash/timeout fail"
            ),
            "invalid_oracle_policy": (
                "exclude reference functions with no checkable or failing oracle"
            ),
        },
        "input_functions": len(rows),
        "valid_functions": len(valid_rows),
        "excluded_invalid_oracle_functions": len(rows) - len(valid_rows),
        "invalid_reference_cases": invalid_rows,
        "summary": summarize(per_case),
        "per_case": per_case,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload["summary"], indent=2))
    print(
        f"Wrote {len(per_case)} valid function rows; "
        f"excluded {len(rows) - len(valid_rows)} to {args.output}"
    )


if __name__ == "__main__":
    main()
