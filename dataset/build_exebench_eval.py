#!/usr/bin/env python3
"""Build a held-out ExeBench decompilation/structure evaluation set.

This benchmark does not assume that every ExeBench function has an executable
test oracle.  It exports the original dependencies and source so the evaluation
can report contextual compilation, token edit similarity, and control-structure
recovery.  Functional claims should use a benchmark with an independent oracle
or the differential-testing subset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir")
    source.add_argument(
        "--parquet",
        type=Path,
        help="local ExeBench parquet with embedded compiler assembly",
    )
    parser.add_argument("--split", default="test_real_compilable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", choices=("clang", "gcc"), default="clang")
    parser.add_argument("--optimization", choices=("O0", "O1", "O2", "O3"), default="O3")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument(
        "--compiled-jsonl",
        type=Path,
        help="Output generated from the same ExeBench split by 1_get_exebench_asm_llvmir.py",
    )
    args = parser.parse_args()

    if args.parquet:
        dataset = load_dataset(
            "parquet", data_files=str(args.parquet), split="train"
        )
    else:
        dataset = load_dataset(
            args.data_dir, split=args.split, trust_remote_code=True
        )
    source_by_index = {
        index: dataset[index]
        for index in range(args.start, min(len(dataset), args.start + args.limit))
    }
    output: list[dict] = []
    if args.parquet:
        target = f"real_{args.compiler}_x86_{args.optimization}"
        for index, record in source_by_index.items():
            assembly = {
                item["target"]: item["code"] for item in record["asm"]
            }.get(target)
            if not assembly or f'{record["fname"]}:' not in assembly:
                continue
            output.append(
                {
                    "task_id": f"exebench/{args.split}/{index}/{args.compiler}/{args.optimization}",
                    "input_asm_prompt": assembly,
                    "type": args.optimization,
                    "compiler": args.compiler,
                    "c_func": record["func_def"],
                    "c_prefix": record.get("real_deps") or "",
                    "c_test": "",
                    "function_name": record["fname"],
                }
            )
    else:
        if not args.compiled_jsonl:
            parser.error("--compiled-jsonl is required with --data-dir")
        asm_key = f"-{args.optimization}_{args.compiler}"
        with args.compiled_jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                index = int(record["idx"])
                if index not in source_by_index or asm_key not in record:
                    continue
                source_record = source_by_index[index]
                output.append(
                    {
                        "task_id": f"exebench/{args.split}/{index}/{args.compiler}/{args.optimization}",
                        "input_asm_prompt": record[asm_key],
                        "type": args.optimization,
                        "compiler": args.compiler,
                        "c_func": record["function_def"],
                        "c_prefix": source_record.get("real_deps") or "",
                        "c_test": "",
                        "function_name": record["function_name"],
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(output)} held-out examples to {args.output}")


if __name__ == "__main__":
    main()
