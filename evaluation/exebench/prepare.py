#!/usr/bin/env python3
"""Build and validate the canonical ExeBench test_real C/O0--O3 dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from evaluation.exebench.common import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_ROOT,
    DEFAULT_METADATA,
    DEFAULT_REFERENCE_VALIDATION,
    OPTIMIZATIONS,
    compile_function_gas,
    evaluate_source,
    load_test_real,
    normalize_function_source,
    save_json,
    sha256,
)


def build_function(
    pair: tuple[int, dict], *, compiler: str, compiler_label: str, timeout: float
) -> tuple[list[dict] | None, dict | None]:
    index, raw = pair
    function_name = raw["fname"]
    source = raw["real_deps"] + "\n" + normalize_function_source(raw["func_def"])
    rows = []
    try:
        for optimization in OPTIMIZATIONS:
            assembly = compile_function_gas(
                source,
                function_name,
                optimization,
                gcc=compiler,
                timeout=timeout,
            )
            rows.append(
                {
                    "task_id": (
                        f"exebench/test_real/{index:04d}/{optimization}"
                    ),
                    "cluster_id": f"exebench/test_real/{index:04d}",
                    "benchmark_index": index,
                    "type": optimization,
                    "language": "c",
                    "compiler": compiler_label,
                    "function_name": function_name,
                    "input_asm_prompt": assembly,
                    "c_func": raw["func_def"],
                    "c_prefix": raw["real_deps"],
                    "c_test": "",
                    "func_head": raw["func_head"],
                    "func_head_types": raw["func_head_types"],
                    "signature": raw["signature"],
                    "real_io_pairs": raw["real_io_pairs"],
                    "real_exe_wrapper": raw["real_exe_wrapper"],
                    "real_iospec": raw["real_iospec"],
                    "source_path": raw["path"],
                }
            )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return None, {
            "benchmark_index": index,
            "function_name": function_name,
            "stage": "assembly",
            "error_type": type(exc).__name__,
            "error": str(exc)[-4000:],
        }
    return rows, None


def validate_reference(
    rows: list[dict], *, c_compiler: str, cxx_compiler: str, timeout: float
) -> tuple[dict, dict | None]:
    row = rows[0]
    result = evaluate_source(
        row,
        row["c_func"],
        gcc=c_compiler,
        gxx=cxx_compiler,
        compile_timeout=timeout,
    ).to_dict()
    record = {
        "cluster_id": row["cluster_id"],
        "benchmark_index": row["benchmark_index"],
        "function_name": row["function_name"],
        **result,
    }
    if result["all_tests_pass"]:
        return record, None
    return record, {
        "benchmark_index": row["benchmark_index"],
        "function_name": row["function_name"],
        "stage": "reference_execution",
        "compile_stage": result["compile_stage"],
        "tests_passed": result["tests_passed"],
        "tests_total": result["tests_total"],
        "error": result["stderr"][-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--reference-validation",
        type=Path,
        default=DEFAULT_REFERENCE_VALIDATION,
    )
    parser.add_argument(
        "--cc",
        "--gcc",
        dest="cc",
        default="gcc",
        help="C compiler used for GAS generation and candidate validation",
    )
    parser.add_argument(
        "--cxx",
        default=None,
        help="C++ compiler used to build the official wrapper",
    )
    parser.add_argument(
        "--validation-cc",
        default=None,
        help=(
            "reference C compiler; defaults to --cc. Set to gcc to isolate "
            "Clang input-codegen effects under the common execution oracle"
        ),
    )
    parser.add_argument(
        "--validation-cxx",
        default=None,
        help="official-wrapper compiler; defaults to --cxx",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--compile-timeout", type=float, default=30.0)
    parser.add_argument(
        "--skip-reference-validation",
        action="store_true",
        help="debug only: do not establish the executable common scope",
    )
    args = parser.parse_args()
    compiler_label = Path(args.cc).name
    if compiler_label.startswith("clang"):
        compiler_label = "clang"
    elif compiler_label.startswith("gcc"):
        compiler_label = "gcc"
    cxx = args.cxx or ("clang++" if compiler_label == "clang" else "g++")
    validation_cc = args.validation_cc or args.cc
    validation_cxx = args.validation_cxx or cxx

    raw = load_test_real(args.dataset_root)
    if len(raw) != 2134:
        raise ValueError(f"expected 2,134 test_real rows, observed {len(raw)}")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        outcomes = list(
            tqdm(
                executor.map(
                    lambda item: build_function(
                        item,
                        compiler=args.cc,
                        compiler_label=compiler_label,
                        timeout=args.compile_timeout,
                    ),
                    enumerate(raw),
                ),
                total=len(raw),
                desc="Building ExeBench O0-O3 GAS",
            )
        )
    groups = [rows for rows, _ in outcomes if rows is not None]
    failures = [error for _, error in outcomes if error is not None]

    validation: list[dict] = []
    if not args.skip_reference_validation:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            checked = list(
                tqdm(
                    executor.map(
                        lambda rows: validate_reference(
                            rows,
                            c_compiler=validation_cc,
                            cxx_compiler=validation_cxx,
                            timeout=args.compile_timeout,
                        ),
                        groups,
                    ),
                    total=len(groups),
                    desc="Validating reference I/O",
                )
            )
        validation = [record for record, _ in checked]
        failures.extend(error for _, error in checked if error is not None)
        valid_ids = {
            record["cluster_id"]
            for record in validation
            if record["all_tests_pass"]
        }
        groups = [rows for rows in groups if rows[0]["cluster_id"] in valid_ids]

    metadata = [
        {
            "benchmark_index": group[0]["benchmark_index"],
            "function_name": group[0]["function_name"],
            "real_io_pairs": group[0]["real_io_pairs"],
            "real_exe_wrapper": group[0]["real_exe_wrapper"],
            "real_iospec": group[0]["real_iospec"],
        }
        for group in groups
    ]
    model_keys = {
        "task_id",
        "cluster_id",
        "benchmark_index",
        "type",
        "language",
        "compiler",
        "function_name",
        "input_asm_prompt",
        "c_func",
        "c_prefix",
        "c_test",
        "func_head",
        "func_head_types",
        "signature",
        "source_path",
    }
    rows = [
        {key: value for key, value in row.items() if key in model_keys}
        for group in groups
        for row in group
    ]
    counts = Counter(row["type"] for row in rows)
    if any(counts[opt] != len(groups) for opt in OPTIMIZATIONS):
        raise RuntimeError(f"unbalanced optimization levels: {dict(counts)}")
    save_json(args.output, rows)
    save_json(args.metadata, metadata)
    save_json(
        args.reference_validation,
        {
            "benchmark": "ExeBench v1.01 test_real",
            "protocol": (
                f"reference C compiled by {validation_cc} at O0; "
                "official real_exe_wrapper; "
                "all 10 provided real_io_pairs must match"
            ),
            "total_raw_functions": len(raw),
            "valid_functions": len(groups),
            "excluded_functions": len(raw) - len(groups),
            "per_case": validation,
            "failures": failures,
        },
    )
    archive = args.dataset_root / "test_real.tar.gz"
    compiler_version = subprocess.run(
        [args.cc, "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()[0]
    manifest = {
        "benchmark": "ExeBench v1.01 test_real",
        "source": str(archive),
        "source_sha256": sha256(archive),
        "scope": {
            "language": "C",
            "excluded_language": "C++",
            "architecture": "x86-64",
            "compiler": args.cc,
            "cxx_compiler": cxx,
            "validation_c_compiler": validation_cc,
            "validation_cxx_compiler": validation_cxx,
            "optimization_levels": list(OPTIMIZATIONS),
            "input_representation": (
                f"{compiler_label} -S target-function GAS"
            ),
            "reference_filter": (
                "assembly available at all four levels and reference passes "
                "all 10 official real I/O pairs"
            ),
        },
        "raw_functions": len(raw),
        "valid_functions": len(groups),
        "instances": len(rows),
        "instances_by_optimization": dict(sorted(counts.items())),
        "compiler_version": compiler_version,
        "reference_validation": str(args.reference_validation),
        "execution_metadata": str(args.metadata),
        "failures": failures,
    }
    save_json(args.output.with_suffix(".manifest.json"), manifest)
    print(
        f"Wrote {len(rows)} rows ({len(groups)} C functions x 4 levels) "
        f"to {args.output}; excluded {len(raw) - len(groups)} functions"
    )


if __name__ == "__main__":
    main()
