#!/usr/bin/env python3
"""Compile and functionally compare outputs from ``run_binary_demo.sh``."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


METHOD_FILES = {
    "DecIR": "decir.json",
    "LLM4Decompile": "llm4decompile.json",
    "sc2dec": "sccdec.json",
    "SLaDe": "slade.json",
    "Nova": "nova.json",
}

PREAMBLE = r"""
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <limits.h>

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
"""

CHECKSUM_HARNESS = r"""
static int reference_checksum(const uint8_t *data, int length) {
    int total = 0;
    for (int i = 0; i < length; ++i) {
        total = (total * 33) ^ data[i];
    }
    return total;
}

int main(void) {
    uint8_t fixed[][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {1, 2, 3, 4, 0, 0, 0, 0},
        {255, 128, 64, 32, 16, 8, 4, 2},
        {7, 11, 13, 17, 19, 23, 29, 31}
    };
    int lengths[] = {0, 4, 8, 8};
    for (int i = 0; i < 4; ++i) {
        int expected = reference_checksum(fixed[i], lengths[i]);
        int actual = checksum(fixed[i], lengths[i]);
        if (actual != expected) {
            return 10 + i;
        }
    }
    uint8_t generated[32];
    uint32_t state = 0x12345678u;
    for (int round = 0; round < 64; ++round) {
        for (int i = 0; i < 32; ++i) {
            state = state * 1664525u + 1013904223u;
            generated[i] = (uint8_t)(state >> 24);
        }
        int length = round % 33;
        if (checksum(generated, length) !=
            reference_checksum(generated, length)) {
            return 20 + round;
        }
    }
    return 0;
}
"""

CLAMP_HARNESS = r"""
static int reference_clamp(int value, int low, int high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

int main(void) {
    int cases[][3] = {
        {-10, 0, 5},
        {3, 0, 5},
        {9, 0, 5},
        {0, 0, 0},
        {INT_MIN, -4, 9},
        {INT_MAX, -4, 9},
        {-2, -5, -1},
        {-8, -5, -1}
    };
    for (int i = 0; i < 8; ++i) {
        int value = cases[i][0];
        int low = cases[i][1];
        int high = cases[i][2];
        if (clamp(value, low, high) != reference_clamp(value, low, high)) {
            return 10 + i;
        }
    }
    return 0;
}
"""

HARNESSES = {"checksum": CHECKSUM_HARNESS, "clamp": CLAMP_HARNESS}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    command: list[str], *, timeout: float
) -> tuple[int | None, str, str, bool]:
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as exc:
        return None, exc.stdout or "", exc.stderr or "", True


def evaluate_source(source: str, function_name: str) -> dict[str, Any]:
    harness = HARNESSES[function_name]
    with tempfile.TemporaryDirectory(prefix="binary-demo-compare-") as tmp:
        tmpdir = Path(tmp)
        candidate_path = tmpdir / "candidate.c"
        object_path = tmpdir / "candidate.o"
        combined_path = tmpdir / "combined.c"
        executable_path = tmpdir / "combined"
        candidate_path.write_text(PREAMBLE + "\n" + source, encoding="utf-8")
        combined_path.write_text(
            PREAMBLE + "\n" + source + "\n" + harness, encoding="utf-8"
        )

        source_rc, _, source_stderr, source_timeout = run(
            [
                "gcc",
                "-std=gnu11",
                "-O0",
                "-c",
                str(candidate_path),
                "-o",
                str(object_path),
            ],
            timeout=10,
        )
        source_compiles = source_rc == 0

        harness_rc, _, harness_stderr, harness_timeout = run(
            [
                "gcc",
                "-std=gnu11",
                "-O0",
                str(combined_path),
                "-o",
                str(executable_path),
            ],
            timeout=10,
        )
        harness_compiles = harness_rc == 0
        run_rc = None
        run_stdout = ""
        run_stderr = ""
        run_timeout = False
        if harness_compiles:
            run_rc, run_stdout, run_stderr, run_timeout = run(
                [str(executable_path)], timeout=2
            )
        outcome = {
            "source_compiles": source_compiles,
            "source_compile_returncode": source_rc,
            "source_compile_timeout": source_timeout,
            "source_compile_stderr": source_stderr[-2000:],
            "harness_compiles": harness_compiles,
            "harness_compile_returncode": harness_rc,
            "harness_compile_timeout": harness_timeout,
            "harness_compile_stderr": harness_stderr[-2000:],
            "tests_pass": harness_compiles and run_rc == 0,
            "test_returncode": run_rc,
            "test_timeout": run_timeout,
            "test_stdout": run_stdout[-1000:],
            "test_stderr": run_stderr[-1000:],
        }
        outcome["diagnosis"] = diagnose(outcome, function_name)
        return outcome


def diagnose(outcome: dict[str, Any], function_name: str) -> str:
    if outcome["tests_pass"]:
        return "passed every deterministic test"
    if not outcome["source_compiles"]:
        return "generated function is not valid compilable GNU C"
    if not outcome["harness_compiles"]:
        return "predicted function signature is incompatible with the reference API"
    if outcome["test_timeout"]:
        return "generated function exceeded the two-second execution limit"
    returncode = outcome["test_returncode"]
    if isinstance(returncode, int) and returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
        return f"generated function terminated with {name}"
    if function_name == "checksum" and isinstance(returncode, int):
        if 10 <= returncode <= 13:
            return f"failed fixed checksum test vector {returncode - 10}"
        if returncode >= 20:
            return f"failed generated checksum round {returncode - 20}"
    if function_name == "clamp" and isinstance(returncode, int):
        return f"failed clamp boundary case {returncode - 10}"
    return f"test executable exited with code {returncode}"


def status(mark: bool) -> str:
    return "PASS" if mark else "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/binary_demo"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("examples/sample_binary.c"),
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("results/binary_demo/sample_binary"),
    )
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    generated_dir = results_dir / "generated_sources"
    generated_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.reference, results_dir / "reference.c")

    comparison: dict[str, Any] = {}
    sources_by_method: dict[str, dict[str, str]] = {}
    for method, filename in METHOD_FILES.items():
        payload = json.loads(
            (results_dir / filename).read_text(encoding="utf-8")
        )
        rows = {
            row["function_name"]: row
            for row in payload["functions"]
        }
        method_dir = generated_dir / method.lower()
        method_dir.mkdir(parents=True, exist_ok=True)
        comparison[method] = {}
        sources_by_method[method] = {}
        for function_name in ("checksum", "clamp"):
            source = rows[function_name]["source"]
            sources_by_method[method][function_name] = source
            (method_dir / f"{function_name}.c").write_text(
                source + "\n", encoding="utf-8"
            )
            comparison[method][function_name] = evaluate_source(
                source, function_name
            )
        if method == "sc2dec":
            comparison[method]["self_context_used"] = {
                name: rows[name]["self_context_used"]
                for name in ("checksum", "clamp")
            }

    compiler = subprocess.run(
        ["gcc", "--version"],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    ).stdout.splitlines()[0]
    reference_rc, _, reference_stderr, reference_timeout = run(
        [str(args.binary.resolve())], timeout=2
    )
    report = {
        "protocol": {
            "source": str(args.reference.resolve()),
            "source_sha256": sha256(args.reference),
            "binary": str(args.binary.resolve()),
            "binary_sha256": sha256(args.binary),
            "compile_command": (
                "gcc -O3 -fno-inline -fno-omit-frame-pointer "
                "examples/sample_binary.c -o "
                "results/binary_demo/sample_binary"
            ),
            "compiler": compiler,
            "reference_binary_returncode": reference_rc,
            "reference_binary_timeout": reference_timeout,
            "reference_binary_stderr": reference_stderr[-1000:],
            "test_timeout_seconds": 2,
        },
        "results": comparison,
    }
    (results_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Binary decompilation demo comparison",
        "",
        "The reference was compiled with GCC `-O3 -fno-inline "
        "-fno-omit-frame-pointer`. `Compile` checks each generated function "
        "alone; `Tests` compiles it with a hidden deterministic harness and "
        "executes it with a two-second timeout.",
        "",
        "| Method | checksum compile | checksum tests | clamp compile | clamp tests |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHOD_FILES:
        checksum = comparison[method]["checksum"]
        clamp = comparison[method]["clamp"]
        lines.append(
            f"| {method} | {status(checksum['source_compiles'])} | "
            f"{status(checksum['tests_pass'])} | "
            f"{status(clamp['source_compiles'])} | "
            f"{status(clamp['tests_pass'])} |"
        )
    lines.extend(["", "## Generated source", ""])
    for method in METHOD_FILES:
        lines.append(f"### {method}")
        lines.append("")
        for function_name in ("checksum", "clamp"):
            outcome = comparison[method][function_name]
            lines.extend(
                [
                    f"#### `{function_name}`",
                    "",
                    f"Compile: **{status(outcome['source_compiles'])}**; "
                    f"tests: **{status(outcome['tests_pass'])}**"
                    + (
                        "; execution timed out"
                        if outcome["test_timeout"]
                        else (
                            f"; test exit code: {outcome['test_returncode']}"
                            if outcome["harness_compiles"]
                            else "; harness did not compile"
                        )
                    ),
                    "",
                    f"Diagnosis: {outcome['diagnosis']}.",
                    "",
                    "```c",
                    sources_by_method[method][function_name],
                    "```",
                    "",
                ]
            )
    (results_dir / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines[: 7 + len(METHOD_FILES)]))
    print(f"\nWrote comparison.json and comparison.md to {results_dir}")


if __name__ == "__main__":
    main()
