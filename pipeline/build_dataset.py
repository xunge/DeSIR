#!/usr/bin/env python3
"""Compile source functions and build DeSIR grounding-aware JSON data.

Input is a JSON list (or JSONL file) with at least ``id``, ``source``, and
``function_name``.  ``tests`` and ``prefix`` are retained for evaluation.
Each source is compiled at O0--O3 and disassembled with objdump; no model is
needed for this stage.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from ground_binary import build_grounding


def load_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    return payload if isinstance(payload, list) else payload["rows"]


def compile_object(source: str, compiler: str, optimization: str, work: Path) -> Path:
    source_path = work / "source.c"
    object_path = work / f"source-{optimization}.o"
    source_path.write_text(source, encoding="utf-8")
    command = [compiler, f"-{optimization}", "-fno-inline", "-fno-omit-frame-pointer",
               "-g0", "-c", str(source_path), "-o", str(object_path)]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"{compiler} {optimization} failed:\n{proc.stderr[-4000:]}")
    return object_path


def build(rows: list[dict], compiler: str, max_rows: int | None) -> list[dict]:
    output = []
    for index, row in enumerate(rows[:max_rows]):
        source = row.get("source", row.get("c_func", row.get("func", "")))
        function = row.get("function_name", row.get("func_name", "func0"))
        if not source.strip():
            raise ValueError(f"row {index} has no source/function body")
        with tempfile.TemporaryDirectory(prefix="desir-build-") as temp:
            work = Path(temp)
            for optimization in ("O0", "O1", "O2", "O3"):
                object_path = compile_object(source, compiler, optimization, work)
                grounding = build_grounding(object_path, function, 32, 32)
                output.append({
                    "id": str(row.get("id", row.get("task_id", index))),
                    "task_id": f"{row.get('id', row.get('task_id', index))}/{optimization}",
                    "type": optimization,
                    "function_name": grounding["function"],
                    "source": source,
                    "c_func": row.get("c_func", source),
                    "c_test": row.get("tests", row.get("c_test", "")),
                    "c_prefix": row.get("prefix", row.get("c_prefix", "")),
                    "assembly": grounding["assembly"],
                    "grounding": grounding["grounding"],
                    "compiler": compiler,
                })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="clang")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if shutil.which(args.compiler) is None:
        parser.error(f"compiler not found: {args.compiler}")
    rows = build(load_rows(args.input), args.compiler, args.max_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} optimization rows to {args.output}")


if __name__ == "__main__":
    main()
