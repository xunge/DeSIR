"""Prompt and output utilities shared by the minimal DeSIR pipeline."""

from __future__ import annotations

import json
import re
from typing import Mapping, Sequence


def c_literal(value: str) -> str:
    """Return a valid C string literal for an extracted binary string."""

    return json.dumps(value, ensure_ascii=True)


def render_prompt(assembly: str, grounding: Mapping[str, Sequence[Mapping]]) -> str:
    """Render the plain-text prompt used by the released DeSIR checkpoints."""

    lines = ["# This is the assembly code:", assembly.rstrip(), "", "# Grounding information:"]
    for kind, prefix in (("strings", "STR"), ("constants", "CONST"), ("data", "DATA")):
        for item in grounding.get(kind, []):
            index = item.get("index", 0)
            value = item.get("value", item.get("text", ""))
            if kind == "strings":
                value = c_literal(str(value))
            lines.append(f"# {prefix}_{index} = {value}")
    lines.extend(["", "# What is the source code?", ""])
    return "\n".join(lines)


def restore_anchors(source: str, grounding: Mapping[str, Sequence[Mapping]]) -> str:
    """Replace model-visible anchor tokens with their recovered C values."""

    replacements = {}
    for kind, prefix in (("strings", "STR"), ("constants", "CONST"), ("data", "DATA")):
        for item in grounding.get(kind, []):
            token = f"{prefix}_{item.get('index', 0)}"
            if kind == "strings":
                replacements[token] = c_literal(str(item.get("value", "")))
            else:
                replacements[token] = str(item.get("value", item.get("text", "0")))
    for token, value in sorted(replacements.items(), key=lambda pair: -len(pair[0])):
        source = re.sub(rf"\b{re.escape(token)}\b", value, source)
    return source


def extract_c_source(text: str) -> str:
    """Keep the longest fenced C block, or the complete generated text."""

    blocks = re.findall(r"```(?:c|C|cpp|C\+\+)?\s*(.*?)```", text, flags=re.DOTALL)
    return max(blocks, key=len).strip() if blocks else text.strip()
