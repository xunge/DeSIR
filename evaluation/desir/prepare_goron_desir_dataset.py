#!/usr/bin/env python3
"""Prepare HumanEval-Goron rows for DeSIR inference.

DeSIR's evaluator normally rebuilds STR/CONST/DATA-aware assembly prompts from
source.  For Goron experiments the prompt must remain the already-obfuscated
assembly stored in ``humaneval-decompile-goron.json``.  This converter marks the
rows as an existing SIR-compatible prompt set with empty anchor metadata, so
``run_evaluation_desir_vllm.py --string_prompt_mode existing`` uses the Goron
assembly verbatim and still performs the post-generation anchor-restoration
path consistently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="clang")
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    prepared = []
    for row in rows:
        item = dict(row)
        item.setdefault("language", "c")
        item.setdefault("function_name", "func0")
        item["string_refs"] = []
        item["const_refs"] = []
        item["data_refs"] = []
        item["anchor_compiler"] = args.compiler
        item["string_compiler"] = args.compiler
        item["anchor_schema_version"] = "sir_asm_anchor_cross_opt_v1"
        item["anchor_preprocessor"] = "goron-obfuscated-assembly-verbatim"
        item["desir_goron_protocol"] = (
            "use existing Goron-obfuscated assembly prompt verbatim; "
            "no STR/CONST/DATA placeholders are injected"
        )
        prepared.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(prepared, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(prepared)} prepared DeSIR-Goron rows to {args.output}")


if __name__ == "__main__":
    main()
