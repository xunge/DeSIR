#!/usr/bin/env python3
"""LLVM-IR normalization variants used by the DecIR experiments.

``legacy`` reproduces the normalization used for the originally released
models.  It is intentionally kept separate because it removes declarations
and unassigned call instructions (among other syntax) and therefore is not
expected to pass an LLVM verifier.

``safe`` removes debug-only text while retaining instructions, declarations,
optimization flags, calling conventions, and alignment information.  It is the
recommended target for new experiments.
"""

from __future__ import annotations

import re
from pathlib import Path


_DEBUG_ATTACHMENT = re.compile(r",?\s*!dbg\s+!\d+")
_METADATA_DEFINITION = re.compile(r"^\s*!(\d+)\s*=")
_METADATA_REFERENCE = re.compile(r"!(\d+)")
_ATTRIBUTE_REFERENCE = re.compile(r"\s*#\d+")
_OTHER_ANNOTATION = re.compile(r",?\s*![A-Za-z_.]+\s+!\d+")
_ALIGNMENT = re.compile(r",\s*align\s+\d+")
_LOCAL_MARKERS = re.compile(
    r" dso_local| readonly| nonnull| noundef| private unnamed_addr constant|"
    r" local_unnamed_addr"
)
_DEBUG_INTRINSIC = re.compile(
    r"^\s*(?:tail\s+)?call\b.*@llvm\.dbg\.(?:addr|assign|declare|label|value)\b"
)
_DEBUG_DECLARATION = re.compile(
    r"^\s*declare\b.*@llvm\.dbg\.(?:addr|assign|declare|label|value)\b"
)


def normalize_ir_safe(raw_ir: str) -> str:
    """Remove debug-only material without deleting program semantics.

    The transformation is deliberately conservative.  In particular, integer
    wrap/exact flags, fast-math flags, calling conventions, and alignments are
    preserved because they can constrain source-level behavior.  The resulting
    module should remain acceptable to ``llvm-as`` or ``clang -x ir`` when the
    raw module was valid.
    """

    output: list[str] = []
    metadata_records: list[tuple[str, str | None, str]] = []
    definitions: dict[str, str] = {}
    for line in raw_ir.splitlines():
        stripped = line.lstrip()
        if not stripped:
            if output and output[-1] != "":
                output.append("")
            continue
        if stripped.startswith(";"):
            continue
        if stripped.startswith("source_filename"):
            continue
        if stripped.startswith("!"):
            definition = _METADATA_DEFINITION.match(line)
            if definition:
                identifier = definition.group(1)
                definitions[identifier] = line
                metadata_records.append(("numbered", identifier, line))
            elif stripped.startswith("!llvm.dbg.cu"):
                metadata_records.append(("debug-named", None, line))
            else:
                metadata_records.append(("named", None, line))
            continue
        if _DEBUG_INTRINSIC.search(line) or _DEBUG_DECLARATION.search(line):
            continue

        line = _DEBUG_ATTACHMENT.sub("", line).rstrip()
        output.append(line)

    while output and output[-1] == "":
        output.pop()

    # Preserve every metadata node reachable from executable IR or from a
    # non-debug named metadata root.  This retains potentially meaningful
    # metadata such as range/non-null constraints instead of assuming that all
    # metadata is optimization-only.
    root_text = "\n".join(output)
    root_text += "\n" + "\n".join(
        line for kind, _, line in metadata_records if kind == "named"
    )
    worklist = list(_METADATA_REFERENCE.findall(root_text))
    reachable: set[str] = set()
    while worklist:
        identifier = worklist.pop()
        if identifier in reachable or identifier not in definitions:
            continue
        reachable.add(identifier)
        worklist.extend(_METADATA_REFERENCE.findall(definitions[identifier]))

    kept_metadata = [
        line
        for kind, identifier, line in metadata_records
        if kind == "named" or (kind == "numbered" and identifier in reachable)
    ]
    if kept_metadata:
        output.extend(["", *kept_metadata])
    return "\n".join(output) + "\n"


def normalize_ir_legacy(raw_ir: str) -> str:
    """Reproduce the lossy normalization used by the released checkpoints."""

    output: list[str] = []
    for line in raw_ir.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith(
            (";", "source_filename", "target", "declare", "attributes", "!", "call")
        ):
            continue
        line = re.sub(r"\s+", " ", line).strip()
        line = _DEBUG_ATTACHMENT.sub("", line)
        line = _ATTRIBUTE_REFERENCE.sub("", line)
        line = _OTHER_ANNOTATION.sub("", line)
        line = _ALIGNMENT.sub("", line)
        line = _LOCAL_MARKERS.sub("", line)
        output.append(re.sub(r"\s+", " ", line))
    return "\n".join(output) + ("\n" if output else "")


def normalize_ir(raw_ir: str, mode: str) -> str:
    if mode == "raw":
        return raw_ir if raw_ir.endswith("\n") else raw_ir + "\n"
    if mode == "safe":
        return normalize_ir_safe(raw_ir)
    if mode == "legacy":
        return normalize_ir_legacy(raw_ir)
    raise ValueError(f"Unknown IR normalization mode: {mode}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--mode", choices=("raw", "safe", "legacy"), default="safe")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        normalize_ir(args.input.read_text(encoding="utf-8"), args.mode),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
