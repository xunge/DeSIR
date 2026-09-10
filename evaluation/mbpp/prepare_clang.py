#!/usr/bin/env python3
"""Build linked MBPP-Decompile C executables with Clang at O0--O3."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm


PROJECT = Path(__file__).resolve().parents[2]
OPTIMIZATIONS = ("O0", "O1", "O2", "O3")
FUNCTION_HEADER = re.compile(r"^[0-9a-fA-F]+\s+<([^>]+)>:$")


def extract_function(disassembly: str, function_name: str) -> str:
    """Convert one GNU-objdump function to the direct-model representation."""

    lines = disassembly.splitlines()
    start = None
    for index, line in enumerate(lines):
        match = FUNCTION_HEADER.match(line.strip())
        if match and match.group(1) == function_name:
            start = index + 1
            break
    if start is None:
        raise ValueError(f"<{function_name}> is absent from objdump output")
    instructions = [f"{function_name}:"]
    for line in lines[start:]:
        if FUNCTION_HEADER.match(line.strip()) or (
            instructions[1:] and not line.strip()
        ):
            break
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        instruction = fields[-1].split("#", 1)[0].strip()
        if instruction:
            instructions.append(instruction)
    if len(instructions) == 1:
        raise ValueError(f"<{function_name}> contains no instructions")
    return "\n".join(instructions)


def compile_row(
    row: dict,
    *,
    clang: str,
    objdump: str,
    crypto_library: str,
    timeout: float,
) -> tuple[dict | None, dict | None]:
    optimization = str(row["type"])
    if optimization not in OPTIMIZATIONS:
        return None, {
            "task_id": row.get("task_id"),
            "stage": "metadata",
            "error": f"unsupported optimization {optimization!r}",
        }
    source = (
        row.get("c_prefix", "")
        + "\n"
        + row["c_func"]
        + "\n\nint main(void) { return 0; }\n"
    )
    with tempfile.TemporaryDirectory(prefix="mbpp-clang-input-") as tmp:
        root = Path(tmp)
        source_path = root / "function.c"
        executable_path = root / "function"
        source_path.write_text(source, encoding="utf-8")
        try:
            process = subprocess.run(
                [
                    clang,
                    f"-{optimization}",
                    "-Wno-implicit-function-declaration",
                    str(source_path),
                    "-o",
                    str(executable_path),
                    "-lm",
                    crypto_library,
                ],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    f"clang returned {process.returncode}: "
                    f"{process.stderr[-4000:]}"
                )
            process = subprocess.run(
                [
                    objdump,
                    "-d",
                    f"--disassemble={row['function_name']}",
                    str(executable_path),
                ],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    f"objdump returned {process.returncode}: "
                    f"{process.stderr[-4000:]}"
                )
            assembly = extract_function(
                process.stdout, str(row["function_name"])
            )
            unresolved_target = re.search(
                rf"\bcallq?\b[^\n]*<{re.escape(str(row['function_name']))}"
                r"\+0x[0-9a-f]+>",
                assembly,
                flags=re.IGNORECASE,
            )
            if unresolved_target:
                raise ValueError(
                    "linked disassembly still contains an unresolved "
                    f"intra-function call target: {unresolved_target.group(0)}"
                )
        except (
            OSError,
            RuntimeError,
            ValueError,
            subprocess.TimeoutExpired,
        ) as exc:
            return None, {
                "task_id": row.get("task_id"),
                "cluster_id": row.get("cluster_id"),
                "type": optimization,
                "stage": "assembly",
                "error_type": type(exc).__name__,
                "error": str(exc)[-4000:],
            }
    return (
        {
            **row,
            "compiler": "clang",
            "input_asm_prompt": assembly,
        },
        None,
    )


def version(command: str) -> str:
    process = subprocess.run(
        [command, "--version"],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    )
    return process.stdout.splitlines()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT / "decompile-eval/mbpp-decompile-c.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT
            / "decompile-eval/mbpp-decompile-clang-linked-c.json"
        ),
    )
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--objdump", default="objdump")
    parser.add_argument(
        "--crypto-library",
        default="-l:libcrypto.so.3",
        help="Exact OpenSSL ABI used when linking reference executables.",
    )
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    rows = json.loads(args.source.read_text(encoding="utf-8"))
    expected = Counter({optimization: 974 for optimization in OPTIMIZATIONS})
    observed = Counter(row["type"] for row in rows)
    if len(rows) != 3896 or observed != expected:
        parser.error(
            "the source must contain 3,896 C rows and 974 rows per level; "
            f"observed {len(rows)} rows and {dict(observed)}"
        )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        outcomes = list(
            tqdm(
                executor.map(
                    lambda row: compile_row(
                        row,
                        clang=args.clang,
                        objdump=args.objdump,
                        crypto_library=args.crypto_library,
                        timeout=args.timeout,
                    ),
                    rows,
                ),
                total=len(rows),
                desc="Building MBPP Clang inputs",
            )
        )
    compiled = [row for row, _ in outcomes if row is not None]
    failures = [failure for _, failure in outcomes if failure is not None]
    if failures:
        failed_clusters = {failure.get("cluster_id") for failure in failures}
        compiled = [
            row for row in compiled if row.get("cluster_id") not in failed_clusters
        ]
    counts = Counter(row["type"] for row in compiled)
    if len(set(counts.values())) > 1:
        raise RuntimeError(f"unbalanced Clang scope: {dict(counts)}")
    call_lines = [
        line
        for row in compiled
        for line in row["input_asm_prompt"].splitlines()
        if re.search(r"\bcallq?\b", line)
    ]
    named_call_count = sum(
        bool(re.search(r"<[^>]+>", line)) for line in call_lines
    )
    if named_call_count != len(call_lines):
        raise RuntimeError(
            "linked input audit found calls without resolved symbols: "
            f"{named_call_count}/{len(call_lines)} calls are named"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(compiled, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "benchmark": "MBPP-Decompile",
        "language": "C",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "compiler": args.clang,
        "compiler_version": version(args.clang),
        "disassembler": args.objdump,
        "disassembler_version": version(args.objdump),
        "input_representation": (
            "Clang linked executable, GNU objdump target function, "
            "addresses and bytes removed"
        ),
        "link_driver": "synthetic int main(void) { return 0; }",
        "link_libraries": ["-lm", args.crypto_library],
        "call_target_audit": {
            "call_count": len(call_lines),
            "named_call_count": named_call_count,
            "unresolved_func_offset_call_count": 0,
        },
        "optimization_levels": list(OPTIMIZATIONS),
        "instances": len(compiled),
        "underlying_clusters": len(
            {row["cluster_id"] for row in compiled}
        ),
        "instances_by_optimization": dict(sorted(counts.items())),
        "failures": failures,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Wrote {len(compiled)} rows to {args.output}; "
        f"{len(failures)} assembly failures"
    )


if __name__ == "__main__":
    main()
