#!/usr/bin/env python3
"""Convert grounding-aware JSON rows into instruction-tuning JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import render_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            prompt = render_prompt(row["assembly"], row.get("grounding", {}))
            completion = row.get("source", row.get("c_func", "")).strip()
            if not completion:
                continue
            handle.write(json.dumps({
                "id": row.get("task_id", row.get("id")),
                "prompt": prompt,
                "completion": completion,
                "type": row.get("type"),
            }, ensure_ascii=False) + "\n")
    print(f"wrote instruction data to {args.output}")


if __name__ == "__main__":
    main()
