#!/usr/bin/env python3
"""Shared, dependency-light helpers for binary decompilation examples.

The examples intentionally accept symbol names rather than attempting
function discovery.  A requested name must therefore be present in an x86-64
ELF symbol table (or dynamic symbol table) and understood by GNU objdump.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DIRECT_PREFIX = "# This is the assembly code:\n"
DIRECT_SUFFIX = "\n# What is the source code?\n"
FENCE_RE = re.compile(r"```(?:c|cpp|c\+\+)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
HEADER_RE = re.compile(r"^\s*[0-9a-fA-F]+\s+<([^>]+)>:\s*$")
INSTRUCTION_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s*(.*?)\s*$")
C_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class BinaryFunction:
    """Two objdump views of one function plus model-oriented adapters."""

    name: str
    raw_objdump: str
    assembly: str
    slade_gas: str


def run_checked(command: list[str], *, timeout: float = 30.0) -> str:
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"required executable is absent: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command timed out: {' '.join(command)}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip()[-2000:] or proc.stdout.strip()[-2000:]
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(command)}\n{detail}"
        )
    return proc.stdout


def validate_binary(binary: Path) -> Path:
    binary = binary.expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"binary does not exist: {binary}")
    header = run_checked(["readelf", "-h", str(binary)])
    if "ELF64" not in header or "Advanced Micro Devices X86-64" not in header:
        raise ValueError(
            "these checkpoints expect an ELF64 x86-64 binary; "
            f"unsupported input: {binary}"
        )
    return binary


def _function_block(objdump_text: str, function_name: str) -> list[str]:
    """Return the exact symbol's objdump block, excluding file/section headers."""

    lines = objdump_text.splitlines()
    start = None
    for index, line in enumerate(lines):
        match = HEADER_RE.match(line)
        if match and match.group(1) == function_name:
            start = index
            break
    if start is None:
        raise ValueError(
            f"function symbol {function_name!r} was not found; "
            "use an unstripped symbol name as printed by `nm -n BINARY`"
        )
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if not lines[index].strip():
            end = index
            break
        # Be defensive if this objdump version does not insert a blank line.
        match = HEADER_RE.match(lines[index])
        if match:
            end = index
            break
    return lines[start:end]


def _objdump(
    binary: Path, function_name: str, *, show_raw_bytes: bool
) -> list[str]:
    command = [
        "objdump",
        "-d",
        "-M",
        "att",
        f"--disassemble={function_name}",
    ]
    if not show_raw_bytes:
        command.append("--no-show-raw-insn")
    command.append(str(binary))
    return _function_block(run_checked(command), function_name)


def _clean_instructions(no_bytes_block: list[str]) -> list[tuple[int, str]]:
    instructions: list[tuple[int, str]] = []
    for line in no_bytes_block[1:]:
        match = INSTRUCTION_RE.match(line)
        if not match:
            continue
        instruction = match.group(2).strip()
        if instruction:
            instructions.append((int(match.group(1), 16), instruction))
    if not instructions:
        raise ValueError("objdump returned the symbol but no instructions")
    return instructions


def _canonical_raw_objdump(raw_block: list[str]) -> str:
    return "\n".join(["0000000000000000 <func0>:", *raw_block[1:]])


def _canonical_direct_assembly(
    instructions: list[tuple[int, str]], function_name: str
) -> str:
    self_reference = re.compile(
        r"<" + re.escape(function_name) + r"(?=[+>])"
    )
    body = [
        self_reference.sub("<func0", text)
        for _, text in instructions
    ]
    return "<func0>:\n" + "\n".join(body)


def _external_target(annotation: str) -> str | None:
    """Map an objdump target annotation to a GAS-ish external symbol."""

    annotation = annotation.split("+", 1)[0]
    if annotation.endswith("@plt"):
        annotation = annotation[:-4] + "@PLT"
    return annotation if annotation else None


def _slade_gas(instructions: list[tuple[int, str]]) -> str:
    """Adapt disassembled AT&T instructions to SLaDe's compiler-GAS envelope.

    Original source-level directives and labels do not survive compilation.
    We reconstruct local branch labels and wrap the instructions in the
    envelope used by SLaDe.  This is necessarily an out-of-distribution adapter
    for an arbitrary binary, and every SLaDe result records that fact.
    """

    addresses = {address for address, _ in instructions}
    branch_targets: set[int] = set()
    direct_target_re = re.compile(
        r"^(?P<op>(?:j\w+|loop\w*|callq?|call))\s+"
        r"(?P<addr>(?:0x)?[0-9a-fA-F]+)"
        r"(?:\s+<(?P<annotation>[^>]+)>)?"
    )
    for _, text in instructions:
        match = direct_target_re.match(text)
        if not match:
            continue
        target = int(match.group("addr"), 16)
        if target in addresses and not match.group("op").startswith("call"):
            branch_targets.add(target)

    labels = {
        address: f".Lbinary_{index}"
        for index, address in enumerate(sorted(branch_targets))
    }
    body: list[str] = []
    for address, text in instructions:
        if address in labels:
            body.append(labels[address] + ":")
        text = text.split("#", 1)[0].strip()
        match = direct_target_re.match(text)
        if match:
            target = int(match.group("addr"), 16)
            replacement = None
            if target in labels:
                replacement = labels[target]
            elif match.group("annotation"):
                replacement = _external_target(match.group("annotation"))
            if replacement:
                text = (
                    match.group("op")
                    + " "
                    + replacement
                    + text[match.end() :]
                )
        body.append("  " + text)
    return "\n".join(
        [
            ".globl func0",
            ".type func0, @function",
            "func0:",
            ".cfi_startproc",
            *body,
            ".cfi_endproc",
        ]
    )


def extract_function(binary: Path, function_name: str) -> BinaryFunction:
    """Extract and canonicalize one named x86-64 ELF function."""

    raw_block = _objdump(binary, function_name, show_raw_bytes=True)
    no_bytes_block = _objdump(binary, function_name, show_raw_bytes=False)
    instructions = _clean_instructions(no_bytes_block)
    return BinaryFunction(
        name=function_name,
        raw_objdump=_canonical_raw_objdump(raw_block),
        assembly=_canonical_direct_assembly(instructions, function_name),
        slade_gas=_slade_gas(instructions),
    )


def extract_functions(
    binary: Path, function_names: list[str]
) -> tuple[Path, list[BinaryFunction]]:
    binary = validate_binary(binary)
    return binary, [extract_function(binary, name) for name in function_names]


def extract_c_code(text: str) -> str:
    matches = FENCE_RE.findall(text)
    return (max(matches, key=len) if matches else text).strip()


def restore_function_name(source: str, function_name: str) -> str:
    """Restore a canonical func0 name when the requested symbol is a C name."""

    source = extract_c_code(source)
    if C_IDENTIFIER_RE.fullmatch(function_name):
        return re.sub(r"\bfunc0\b", function_name, source)
    return source


def direct_prompt(assembly: str) -> str:
    return DIRECT_PREFIX + assembly.strip() + DIRECT_SUFFIX


def read_function_names(args: argparse.Namespace) -> list[str]:
    if args.functions is not None:
        names = args.functions
    else:
        text = args.functions_file.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            payload = json.loads(text)
            if not isinstance(payload, list) or not all(
                isinstance(item, str) for item in payload
            ):
                raise ValueError("JSON function list must be an array of strings")
            names = payload
        else:
            names = [
                line.strip()
                for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError("the function list is empty")
    if any("\x00" in name or "\n" in name for name in names):
        raise ValueError("function names may not contain NUL or newline")
    return names


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--binary", type=Path, required=True)
    functions = parser.add_mutually_exclusive_group(required=True)
    functions.add_argument(
        "--functions",
        nargs="+",
        help="ELF symbol names, for example: main helper checksum",
    )
    functions.add_argument(
        "--functions-file",
        type=Path,
        help="One symbol per line, or a JSON string array.",
    )
    parser.add_argument("--output", type=Path, required=True)


def write_results(
    *,
    output: Path,
    method: str,
    binary: Path,
    records: list[dict[str, Any]],
    configuration: dict[str, Any],
) -> None:
    payload = {
        "method": method,
        "binary": str(binary),
        "configuration": configuration,
        "functions": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(output)
    for record in records:
        print(f"\n/* ===== {record['function_name']} ===== */")
        print(record.get("source") or f"/* ERROR: {record.get('error')} */")
    print(f"\nWrote {len(records)} function result(s) to {output}")
