#!/usr/bin/env python3
"""Extract a compact, deterministic grounding record from an ELF/object file.

This is intentionally dependency-free.  It uses binutils output rather than
debug symbols, so the same script can process stripped binaries and compiler
objects.  The record is consumed by :mod:`decompile_binary` and is also useful
for auditing prompts before a model is run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


FUNCTION_RE = re.compile(r"^\s*([0-9a-fA-F]+)\s+<([^>]+)>:\s*$", re.MULTILINE)
STRING_RE = re.compile(r"^\s*([0-9a-fA-F]+)\s+(.*)$")
SYMBOL_RE = re.compile(r"^([0-9a-fA-F]+)\s+([A-Za-z])\s+(.+)$")


def command(args: list[str]) -> str:
    try:
        proc = subprocess.run(args, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"Required tool is not installed: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"{' '.join(args)} failed:\n{exc.stderr[-4000:]}") from exc
    return proc.stdout


def extract_function(disassembly: str, function: str | None) -> tuple[str, str]:
    """Return ``(resolved_name, body)`` for one objdump function block."""

    matches = list(FUNCTION_RE.finditer(disassembly))
    if not matches:
        raise ValueError("objdump did not contain any function symbols")
    selected = None
    for match in matches:
        name = match.group(2).split("@", 1)[0]
        if function is None or name == function:
            selected = match
            break
    if selected is None:
        available = ", ".join(match.group(2) for match in matches[:20])
        raise ValueError(f"function {function!r} not found; available: {available}")
    index = matches.index(selected)
    end = matches[index + 1].start() if index + 1 < len(matches) else len(disassembly)
    return selected.group(2), disassembly[selected.start():end].strip()


def extract_strings(binary: Path, limit: int) -> list[dict]:
    raw = command(["strings", "-a", "-t", "x", str(binary)])
    records = []
    for line in raw.splitlines():
        match = STRING_RE.match(line)
        if not match:
            continue
        value = match.group(2).strip()
        if not value or len(value) > 256:
            continue
        records.append({"index": len(records), "address": "0x" + match.group(1), "value": value})
        if len(records) >= limit:
            break
    return records


def extract_symbols(binary: Path, limit: int) -> list[dict]:
    raw = command(["nm", "-C", "--defined-only", str(binary)])
    records = []
    for line in raw.splitlines():
        match = SYMBOL_RE.match(line.strip())
        if not match:
            continue
        symbol_type = match.group(2)
        if symbol_type not in {"B", "b", "D", "d", "R", "r", "S", "s", "V", "v"}:
            continue
        records.append({
            "index": len(records),
            "address": "0x" + match.group(1),
            "type": symbol_type,
            "value": match.group(3).strip(),
        })
        if len(records) >= limit:
            break
    return records


def build_grounding(binary: Path, function: str | None, max_strings: int, max_symbols: int) -> dict:
    disassembly = command(["objdump", "-d", "-M", "intel", str(binary)])
    resolved, body = extract_function(disassembly, function)
    strings = extract_strings(binary, max_strings)
    symbols = extract_symbols(binary, max_symbols)
    return {
        "schema": "desir-grounding-v1",
        "binary": str(binary.resolve()),
        "function": resolved,
        "assembly": body,
        "grounding": {
            "strings": strings,
            "constants": [],
            "data": symbols,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--function", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-strings", type=int, default=32)
    parser.add_argument("--max-symbols", type=int, default=32)
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error(f"binary/object does not exist: {args.binary}")
    record = build_grounding(args.binary, args.function, args.max_strings, args.max_symbols)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"grounded {record['function']} -> {args.output}")


if __name__ == "__main__":
    main()
