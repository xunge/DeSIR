#!/usr/bin/env python3
"""Random differential testing for the scalar-signature benchmark subset.

The harness intentionally reports its eligibility coverage.  Functions with
pointers, arrays, aggregates, variadic arguments, or void returns are skipped
rather than assigned a misleading semantic-equivalence result.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from common import extract_c_code, split_includes


SCALAR_WORDS = {
    "bool",
    "_Bool",
    "char",
    "signed",
    "unsigned",
    "short",
    "int",
    "long",
    "float",
    "double",
    "size_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
}
QUALIFIERS = {"const", "volatile", "restrict", "static", "inline", "extern"}


def split_parameters(text: str) -> list[str] | None:
    if not text.strip() or text.strip() == "void":
        return []
    if "..." in text:
        return None
    depth = 0
    start = 0
    parts = []
    for index, char in enumerate(text):
        depth += char in "(["
        depth -= char in ")]"
        if char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def scalar_type(declaration: str) -> tuple[str, str] | None:
    if any(symbol in declaration for symbol in ("*", "[", "]", "(", ")")):
        return None
    declaration = re.sub(r"\s+", " ", declaration.strip())
    match = re.search(r"([A-Za-z_]\w*)\s*$", declaration)
    if not match:
        return None
    name = match.group(1)
    type_text = declaration[: match.start()].strip()
    words = [word for word in type_text.split() if word not in QUALIFIERS]
    if not words or any(word not in SCALAR_WORDS for word in words):
        return None
    return type_text, name


def parse_signature(code: str, preferred_name: str | None) -> dict | None:
    if preferred_name:
        name_pattern = re.escape(preferred_name)
    else:
        name_pattern = r"[A-Za-z_]\w*"
    pattern = re.compile(
        rf"(?P<return>[A-Za-z_][\w\s]*?)\s+(?P<name>{name_pattern})\s*"
        rf"\((?P<params>[^{{;}}]*)\)\s*\{{",
        re.MULTILINE,
    )
    match = pattern.search(code)
    if not match:
        return None
    return_type = match.group("return").strip()
    return_words = [word for word in return_type.split() if word not in QUALIFIERS]
    if (
        "*" in return_type
        or not return_words
        or "void" in return_words
        or any(word not in SCALAR_WORDS for word in return_words)
    ):
        return None
    raw_params = split_parameters(match.group("params"))
    if raw_params is None:
        return None
    params = []
    for raw in raw_params:
        parsed = scalar_type(raw)
        if parsed is None:
            return None
        params.append(parsed)
    return {
        "return_type": return_type,
        "name": match.group("name"),
        "params": params,
    }


def rename_function(code: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, code)


def argument_expression(type_text: str, index: int) -> str:
    if "bool" in type_text or "_Bool" in type_text:
        return f"({type_text})(decir_rand() & 1u)"
    if "float" in type_text or "double" in type_text:
        return f"({type_text})(((int)(decir_rand() % 2001u) - 1000) / 17.0)"
    return f"({type_text})((int)(decir_rand() % 2001u) - 1000 + {index})"


def harness(signature: dict, trials: int, seed: int) -> str:
    args = [
        f"a{index}"
        for index, _ in enumerate(signature["params"])
    ]
    declarations = "\n".join(
        f"    {type_text} a{index} = {argument_expression(type_text, index)};"
        for index, (type_text, _) in enumerate(signature["params"])
    )
    call_args = ", ".join(args)
    floating = "float" in signature["return_type"] or "double" in signature["return_type"]
    if floating:
        comparison = (
            "if (!((isnan((double)ref) && isnan((double)got)) || "
            "fabs((double)ref - (double)got) <= "
            "1e-6 * fmax(1.0, fabs((double)ref)))) return 1;"
        )
    else:
        comparison = "if (ref != got) return 1;"
    return f"""
#include <stdint.h>
#include <math.h>
static uint64_t decir_state = UINT64_C({seed or 1});
static uint64_t decir_rand(void) {{
    decir_state ^= decir_state << 13;
    decir_state ^= decir_state >> 7;
    decir_state ^= decir_state << 17;
    return decir_state;
}}
int main(void) {{
  for (int trial = 0; trial < {trials}; ++trial) {{
{declarations}
    {signature['return_type']} ref = decir_reference({call_args});
    {signature['return_type']} got = decir_candidate({call_args});
    {comparison}
  }}
  return 0;
}}
"""


def evaluate(item: dict, compiler: str, trials: int, seed: int, timeout: float) -> dict:
    reference = item["c_func"]
    candidate = extract_c_code(item["output"])
    signature = parse_signature(reference, item.get("function_name"))
    base = {"task_id": item.get("task_id", item.get("id"))}
    if signature is None:
        return {**base, "eligible": False, "reason": "unsupported_signature"}

    includes, (reference, candidate) = split_includes(reference, candidate)
    reference = rename_function(reference, signature["name"], "decir_reference")
    candidate = rename_function(candidate, signature["name"], "decir_candidate")
    program = (
        includes
        + "\n"
        + reference
        + "\n"
        + candidate
        + "\n"
        + harness(signature, trials, seed)
    )
    with tempfile.TemporaryDirectory(prefix="decir-diff-") as tmp:
        source = Path(tmp) / "differential.c"
        executable = Path(tmp) / "differential"
        source.write_text(program, encoding="utf-8")
        try:
            compiled = subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    "-O0",
                    "-fno-strict-aliasing",
                    str(source),
                    "-o",
                    str(executable),
                    "-lm",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if compiled.returncode != 0:
                return {
                    **base,
                    "eligible": True,
                    "compiled": False,
                    "passed": False,
                    "stderr": compiled.stderr[-4000:],
                }
            ran = subprocess.run(
                [str(executable)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                **base,
                "eligible": True,
                "compiled": True,
                "passed": ran.returncode == 0,
                "run_returncode": ran.returncode,
                "stderr": ran.stderr[-4000:],
            }
        except subprocess.TimeoutExpired as exc:
            return {
                **base,
                "eligible": True,
                "compiled": False,
                "passed": False,
                "timed_out": True,
                "stderr": str(exc),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="clang")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    items = json.loads(args.predictions.read_text(encoding="utf-8"))
    rows = [
        evaluate(item, args.compiler, args.trials, args.seed + index, args.timeout)
        for index, item in enumerate(items)
    ]
    eligible = [row for row in rows if row["eligible"]]
    passed = sum(row.get("passed", False) for row in eligible)
    summary = {
        "total": len(rows),
        "eligible": len(eligible),
        "coverage": round(100 * len(eligible) / len(rows), 2) if rows else 0.0,
        "passed": passed,
        "pass_rate_among_eligible": round(100 * passed / len(eligible), 2)
        if eligible
        else 0.0,
        "trials_per_function": args.trials,
        "seed": args.seed,
    }
    payload = {"summary": summary, "per_case": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
