#!/usr/bin/env python3
"""Build and audit the executable-validated Goron HumanEval dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (
    DATASET,
    GORON_CLANG,
    METRICS,
    NOVA_ROOT,
    SOURCE_DATASET,
    load_rows,
    run,
)


def dataset_complete(path: Path) -> bool:
    try:
        rows = load_rows(path)
    except (OSError, ValueError):
        return False
    return (
        len(rows) == 3936
        and all(row.get("goron", {}).get("reference_test_passed") for row in rows)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DATASET)
    parser.add_argument("--output", type=Path, default=DATASET)
    parser.add_argument("--clang", type=Path, default=GORON_CLANG)
    parser.add_argument("--nova-root", type=Path, default=NOVA_ROOT)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force or not dataset_complete(args.output):
        run(
            [
                sys.executable,
                "dataset/build_goron_humaneval.py",
                "--dataset",
                args.source,
                "--output",
                args.output,
                "--nova-root",
                args.nova_root,
                "--clang",
                args.clang,
                "--workers",
                str(args.workers),
            ]
        )
    else:
        print(f"Reusing validated 3,936-row dataset: {args.output}")

    run(
        [
            sys.executable,
            "evaluation/audit_goron_inputs.py",
            "--dataset",
            args.output,
            "--output",
            METRICS / "humaneval-goron-input-audit.json",
        ]
    )


if __name__ == "__main__":
    main()
