#!/usr/bin/env python3
"""Build reproducible DecIR training/validation ablation datasets.

The script consumes JSONL records produced by
``1_get_exebench_asm_llvmir.py``.  Splits are made by a stable hash of the
function identity before optimization/compiler expansion, so variants of one
function cannot leak between training and validation.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Iterator


DIRECT_PREFIX = "# This is the assembly code:\n"
DIRECT_SUFFIX = "\n# What is the source code?\n"
ALIGN_PREFIX = "# Convert the assembly code to LLVM-IR, then write the equivalent C function.\n"
IR_HEADER = "# Here is the LLVM-IR code:\n"
C_HEADER = "\n# And here is the equivalent C function:\n"


def input_paths(specs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for spec in specs:
        path = Path(spec)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.jsonl")))
        else:
            matches = [Path(match) for match in sorted(glob.glob(spec))]
            paths.extend(matches or [path])
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input files not found: {missing}")
    return paths


def read_jsonl(paths: Iterable[Path]) -> Iterator[dict]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_number}: {exc}") from exc


def identity(record: dict) -> str:
    if "idx" in record:
        return str(record["idx"])
    material = "\0".join(
        str(record.get(key, "")) for key in ("function_name", "function_def", "formatted_code")
    )
    return hashlib.sha256(material.encode()).hexdigest()


def is_validation(record: dict, fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{identity(record)}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < fraction


def select_partition(record: dict, partition: str, fraction: float, seed: int) -> bool:
    if partition == "all":
        return True
    validation = is_validation(record, fraction, seed)
    return validation if partition == "validation" else not validation


def ir_field(opt: str, variant: str) -> str:
    suffix = {"legacy": "ll", "safe": "ll_safe", "raw": "ll_raw"}[variant]
    return f"-{opt}_{suffix}"


def available_examples(record: dict, compilers: list[str], ir_variant: str) -> Iterator[dict]:
    source = record["function_def"]
    for opt in ("O0", "O1", "O2", "O3"):
        ir_key = ir_field(opt, ir_variant)
        if ir_key not in record:
            continue
        for compiler in compilers:
            asm_key = f"-{opt}_{compiler}"
            if asm_key not in record:
                continue
            yield {
                "id": identity(record),
                "optimization": opt,
                "compiler": compiler,
                "assembly": record[asm_key],
                "ir": record[ir_key],
                "source_code": source,
            }


def target_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="JSONL files, directories, or globs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--recipe",
        choices=("full", "direct", "direct-repeat", "align-only"),
        default="full",
    )
    parser.add_argument("--ir", choices=("legacy", "safe", "raw", "shuffled-safe"), default="safe")
    parser.add_argument("--compiler", choices=("clang", "gcc", "both"), default="clang")
    parser.add_argument("--partition", choices=("train", "validation", "all"), default="train")
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional Hugging Face tokenizer for exact token-budget matching.",
    )
    args = parser.parse_args()

    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")

    base_ir = "safe" if args.ir == "shuffled-safe" else args.ir
    compilers = ["clang", "gcc"] if args.compiler == "both" else [args.compiler]
    records = [
        record
        for record in read_jsonl(input_paths(args.input))
        if select_partition(
            record, args.partition, args.validation_fraction, args.seed
        )
    ]
    examples = [
        example
        for record in records
        for example in available_examples(record, compilers, base_ir)
    ]
    if not examples:
        raise ValueError(
            "No examples were produced. Check the selected partition and IR fields; "
            "safe/raw fields require regenerating the compiled dataset."
        )

    if args.ir == "shuffled-safe":
        rng = random.Random(args.seed)
        shuffled = [example["ir"] for example in examples]
        rng.shuffle(shuffled)
        if len(shuffled) > 1 and any(
            value == example["ir"] for value, example in zip(shuffled, examples)
        ):
            shuffled = shuffled[1:] + shuffled[:1]
        for example, replacement in zip(examples, shuffled):
            example["ir"] = replacement

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    rows: list[dict] = []
    direct_budget = 0
    full_budget = 0
    for example in examples:
        direct = {
            "source": DIRECT_PREFIX + example["assembly"] + DIRECT_SUFFIX,
            "target": example["source_code"],
            "category": "direct",
            "metadata": {
                key: example[key] for key in ("id", "optimization", "compiler")
            },
        }
        align = {
            "source": ALIGN_PREFIX + example["assembly"],
            "target": IR_HEADER + example["ir"] + C_HEADER + example["source_code"],
            "category": f"align-{args.ir}",
            "metadata": {
                key: example[key] for key in ("id", "optimization", "compiler")
            },
        }
        direct_budget += target_tokens(direct["target"], tokenizer)
        full_budget += target_tokens(direct["target"], tokenizer) + target_tokens(
            align["target"], tokenizer
        )
        if args.recipe in ("full", "direct", "direct-repeat"):
            rows.append(direct)
        if args.recipe in ("full", "align-only"):
            rows.append(align)

    if args.recipe == "direct-repeat":
        # Repeat direct examples until their supervised-target budget matches
        # the full dual-task dataset.  This controls for the extra number of
        # target tokens seen by the auxiliary-supervision condition.
        base_rows = list(rows)
        current_budget = direct_budget
        cursor = 0
        while current_budget < full_budget:
            row = dict(base_rows[cursor % len(base_rows)])
            row["metadata"] = dict(row["metadata"], repeat=cursor // len(base_rows) + 1)
            rows.append(row)
            current_budget += target_tokens(row["target"], tokenizer)
            cursor += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "inputs": [str(path) for path in input_paths(args.input)],
        "output": str(args.output),
        "recipe": args.recipe,
        "ir": args.ir,
        "compiler": args.compiler,
        "partition": args.partition,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "functions": len(records),
        "expanded_examples": len(examples),
        "rows": len(rows),
        "direct_target_tokens": direct_budget,
        "full_target_tokens": full_budget,
        "token_counting": "huggingface" if tokenizer else "whitespace-approximation",
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
