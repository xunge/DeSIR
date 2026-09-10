#!/usr/bin/env python3
"""Import a Decompile-Bench-Eval JSON file into the shared experiment schema.

The upstream files store one row per optimization level and use a monotonically
increasing ``index`` for every row. This importer gives the four optimization
variants of one source translation a common task prefix. For MBPP's parallel
C and C++ translations, ``cluster_id`` also groups both languages so
uncertainty estimates resample underlying algorithms rather than treating
correlated binaries as independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


EXPECTED_OPTS = ("O0", "O1", "O2", "O3")
REQUIRED_FIELDS = {
    "index",
    "func_name",
    "func_dep",
    "func",
    "test",
    "opt",
    "language",
    "asm",
}


def normalized_source(source: str) -> str:
    return re.sub(r"\s+", "", source)


def convert(rows: list[dict], benchmark: str) -> list[dict]:
    if len(rows) % len(EXPECTED_OPTS):
        raise ValueError(
            f"{len(rows)} rows cannot be partitioned into {len(EXPECTED_OPTS)} "
            "optimization variants"
        )

    converted: list[dict] = []
    per_language_programs = {"c": 0, "cpp": 0}
    for start in range(0, len(rows), len(EXPECTED_OPTS)):
        group = rows[start : start + len(EXPECTED_OPTS)]
        missing = [sorted(REQUIRED_FIELDS - row.keys()) for row in group]
        if any(missing):
            raise ValueError(f"missing fields near row {start}: {missing}")
        observed_opts = tuple(row["opt"] for row in group)
        if observed_opts != EXPECTED_OPTS:
            raise ValueError(
                f"rows {start}:{start + len(group)} have optimization order "
                f"{observed_opts}, expected {EXPECTED_OPTS}"
            )
        identity = {
            (row["func_dep"], row["func"], row["test"], row["language"])
            for row in group
        }
        if len(identity) != 1:
            raise ValueError(
                f"rows {start}:{start + len(group)} do not share source/test/language"
            )

        program_index = start // len(EXPECTED_OPTS)
        language = group[0]["language"].lower()
        language_program_index = per_language_programs[language]
        per_language_programs[language] += 1
        for row in group:
            language = row["language"].lower()
            if language not in {"c", "cpp"}:
                raise ValueError(
                    f"unsupported language {row['language']!r} at row {row['index']}"
                )
            converted.append(
                {
                    "task_id": f"{benchmark}/{program_index:04d}/{row['opt']}",
                    # MBPP contains parallel C and C++ translations in the
                    # same per-language order. Cluster both translations and
                    # all optimization variants as one underlying algorithm.
                    "cluster_id": (
                        f"{benchmark}/{language_program_index:04d}"
                    ),
                    "benchmark_index": row["index"],
                    "type": row["opt"],
                    "language": language,
                    "compiler": "g++" if language == "cpp" else "gcc",
                    "function_name": row["func_name"],
                    "input_asm_prompt": row["asm"].strip(),
                    "c_func": row["func"],
                    "c_prefix": row["func_dep"],
                    "c_test": row["test"],
                }
            )
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        default="mbpp-decompile",
        help="stable task-id prefix used in saved predictions and metrics",
    )
    parser.add_argument(
        "--comparison-dataset",
        type=Path,
        help=(
            "optional shared-schema JSON used to audit exact "
            "whitespace-normalized source overlap"
        ),
    )
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    converted = convert(rows, args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(converted, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    counts = Counter(row["language"] for row in converted)
    manifest = {
        "source": str(args.input),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "benchmark": args.benchmark,
        "instances": len(converted),
        "source_translations": len(converted) // len(EXPECTED_OPTS),
        "underlying_clusters": len(
            {row["cluster_id"] for row in converted}
        ),
        "languages": dict(sorted(counts.items())),
        "optimization_levels": dict(
            sorted(Counter(row["type"] for row in converted).items())
        ),
    }
    if args.comparison_dataset:
        comparison_rows = json.loads(
            args.comparison_dataset.read_text(encoding="utf-8")
        )
        imported_sources = {
            normalized_source(row["c_func"]) for row in converted
        }
        comparison_sources = {
            normalized_source(row["c_func"]) for row in comparison_rows
        }
        manifest["source_overlap_audit"] = {
            "comparison_dataset": str(args.comparison_dataset),
            "comparison_sha256": hashlib.sha256(
                args.comparison_dataset.read_bytes()
            ).hexdigest(),
            "matching_rule": "exact after removing all whitespace",
            "imported_unique_sources": len(imported_sources),
            "comparison_unique_sources": len(comparison_sources),
            "overlap": len(imported_sources & comparison_sources),
        }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(converted)} instances "
        f"({len(converted) // len(EXPECTED_OPTS)} programs; "
        f"C={counts['c']}, C++={counts['cpp']}) to {args.output}; "
        f"manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
