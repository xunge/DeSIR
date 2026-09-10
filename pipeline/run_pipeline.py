#!/usr/bin/env python3
"""One-command binary -> grounding -> DeSIR source pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decompile_binary import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--model", default="xunge/desir-1.3b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    result = run(args.binary, args.function, args.model, None, args.output,
                  args.max_new_tokens, 0.0)
    print(json.dumps({"function": result["function"], "source": result["source"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
