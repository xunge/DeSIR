#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import gc
import importlib.util
import json
import math
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import traceback
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from loguru import logger
from tqdm import tqdm


# ``run_evaluation_desir_vllm.py`` lives in ``evaluation/desir``.  Resolve the
# repository root explicitly so the copied open-source layout works regardless
# of the caller's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = PROJECT_ROOT / "evaluation"
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

logger.add(sys.stdout, colorize=False, format="{time} {level} {message}")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

OPT_PROMPTS = {opt: "# This is the assembly code:\n" for opt in ("O0", "O1", "O2", "O3")}
AFTER_PROMPT = "\n# What is the source code?\n"

def _discover_default_sir_preprocessor() -> Path:
    candidates = (
        Path(__file__).resolve().parent / "1_get_exebench_asm_sir.py",
        PROJECT_ROOT / "dataset_sir" / "1_get_exebench_asm_sir.py",
        PROJECT_ROOT / "dataset_const" / "1_get_exebench_asm_sir.py",
        PROJECT_ROOT / "dataset" / "1_get_exebench_asm_sir.py",
        PROJECT_ROOT / "1_get_exebench_asm_sir.py",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


DEFAULT_ANCHOR_PREPROCESSOR = _discover_default_sir_preprocessor()
PREPARED_SCHEMA_VERSION = "sir_asm_anchor_cross_opt_v1"

ANCHOR_KINDS = ("string", "const", "data")
ANCHOR_PREFIX = {
    "string": "STR",
    "const": "CONST",
    "data": "DATA",
}
ANCHOR_TOKEN_RE = re.compile(r"\b(?:STR|CONST|DATA)_\d+\b")

_ANCHOR_PREPROCESSOR = None
_ANCHOR_PREPROCESSOR_PATH = None


def parse_args():
    parser = ArgumentParser(
        description=(
            "Evaluate a DecIR-SIR model with STR/CONST/DATA-aware HumanEval prompts using vLLM."
        )
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--compiler", choices=["clang", "gcc"], default="clang")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--max_num_seqs", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.82)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_total_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--testset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--output_result_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--reuse_outputs", action="store_true")

    # 为兼容旧启动命令保留 string_prompt_mode 这个名字。
    # 现在它实际控制 STR/CONST/DATA 三类 anchor。
    parser.add_argument(
        "--string_prompt_mode",
        choices=["auto", "existing", "rebuild"],
        default="auto",
        help=(
            "auto: reuse a valid prepared STR/CONST/DATA test set if available, "
            "otherwise rebuild; existing: require an already prepared anchor-aware "
            "test set; rebuild: recompile every case and regenerate all anchors."
        ),
    )
    parser.add_argument(
        "--prepared_testset_output",
        type=str,
        default=None,
        help="Optional JSON cache for the prepared STR/CONST/DATA-aware test set.",
    )
    parser.add_argument(
        "--anchor_preprocessor_path",
        type=str,
        default=str(DEFAULT_ANCHOR_PREPROCESSOR),
        help=(
            "Path to the SIR Stage-1 generator 1_get_exebench_asm_sir.py. "
            "Its decompile_bin_to_asm() must return (asm_clean, anchors)."
        ),
    )
    parser.add_argument("--string_threshold", type=int, default=128)
    parser.add_argument("--min_string_length", type=int, default=1)
    parser.add_argument("--max_scan_bytes", type=int, default=1048576)
    parser.add_argument(
        "--clang_bin",
        type=str,
        default=None,
        help="Optional Clang binary override. Default comes from 1_get_exebench_asm_sir.py (normally clang-18).",
    )
    parser.add_argument(
        "--gcc_bin",
        type=str,
        default=None,
        help="Optional GCC binary override. Default comes from 1_get_exebench_asm_sir.py (normally gcc).",
    )
    parser.add_argument(
        "--prompt_workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
        help="Compiler workers used when anchor-aware prompts must be rebuilt.",
    )
    parser.add_argument("--compile_timeout", type=int, default=10)
    parser.add_argument("--run_timeout", type=int, default=10)
    return parser.parse_args()


# =============================================================================
# DecIR-SIR Stage-1 preprocessor
# =============================================================================

def _load_anchor_preprocessor(module_path: Optional[str] = None):
    """
    Load the DecIR-SIR Stage-1 generator 1_get_exebench_asm_sir.py.

    Its real interface is:
        decompile_bin_to_asm(...) -> (asm_clean, anchors)

    anchors is a mixed list with anchor["kind"] in:
        string / const / data
    """
    global _ANCHOR_PREPROCESSOR, _ANCHOR_PREPROCESSOR_PATH

    path = Path(module_path or DEFAULT_ANCHOR_PREPROCESSOR).expanduser().resolve()
    path_text = str(path)

    if _ANCHOR_PREPROCESSOR is not None and _ANCHOR_PREPROCESSOR_PATH == path_text:
        return _ANCHOR_PREPROCESSOR

    if not path.is_file():
        raise FileNotFoundError(f"Anchor preprocessor not found: {path}")

    # The SIR generator imports string_utils as a top-level module. Prefer its
    # own directory, but also add the common DecIR dataset directories so the
    # evaluator can run when string_utils is shared rather than copied.
    import_dirs = (
        path.parent,
        PROJECT_ROOT / "dataset_sir",
        PROJECT_ROOT / "dataset_const",
        PROJECT_ROOT / "dataset_string",
    )
    for import_dir in import_dirs:
        import_text = str(import_dir)
        if import_dir.is_dir() and import_text not in sys.path:
            sys.path.insert(0, import_text)

    spec = importlib.util.spec_from_file_location("decir_dataset_sir_stage1", path_text)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load anchor preprocessor: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not callable(getattr(module, "decompile_bin_to_asm", None)):
        raise AttributeError(
            f"{path} does not define a callable decompile_bin_to_asm()"
        )

    _ANCHOR_PREPROCESSOR = module
    _ANCHOR_PREPROCESSOR_PATH = path_text
    return module


def _configure_sir_preprocessor_toolchain(
    preprocessor,
    clang_bin: Optional[str] = None,
    gcc_bin: Optional[str] = None,
) -> None:
    """Keep HumanEval ASM generation on the same compiler binaries as SIR Stage 1."""
    if clang_bin:
        preprocessor.CLANG_BIN = str(clang_bin)
    if gcc_bin:
        preprocessor.GCC_BIN = str(gcc_bin)


def _compiler_binary_for_preprocessor(preprocessor, compiler: str) -> str:
    if compiler == "clang":
        return str(getattr(preprocessor, "CLANG_BIN", "clang-18"))
    if compiler == "gcc":
        return str(getattr(preprocessor, "GCC_BIN", "gcc"))
    raise ValueError(f"Unsupported compiler: {compiler}")


def _compact_anchor_refs_fallback(
    anchors: Sequence[Mapping[str, object]], kind: str
) -> List[Dict[str, object]]:
    """Fallback matching SIR Stage-1 _compact_anchor_refs()."""
    refs: List[Dict[str, object]] = []

    for anchor in anchors:
        if anchor.get("kind") != kind:
            continue

        if kind == "string":
            refs.append(
                {
                    "id": str(anchor["id"]),
                    "bytes_hex": str(anchor.get("bytes_hex", "")),
                    "value": str(anchor.get("value", anchor.get("display", ""))),
                }
            )
        elif kind == "const":
            refs.append(
                {
                    "id": str(anchor["id"]),
                    "type": str(anchor.get("type", "")),
                    "bytes_hex": str(anchor.get("bytes_hex", "")),
                    "value": anchor.get("value"),
                    "c_literal": str(
                        anchor.get("c_literal", anchor.get("display", ""))
                    ),
                }
            )
        elif kind == "data":
            refs.append(
                {
                    "id": str(anchor["id"]),
                    "type": str(anchor.get("type", "")),
                    "count": int(anchor.get("count", 0)),
                    "bytes_hex": str(anchor.get("bytes_hex", "")),
                    "values": list(anchor.get("values", [])),
                }
            )
        else:
            raise ValueError(f"Unsupported anchor kind: {kind}")

    return refs


def _compact_anchor_refs(preprocessor, anchors, kind: str):
    compact_fn = getattr(preprocessor, "_compact_anchor_refs", None)
    if callable(compact_fn):
        return compact_fn(anchors, kind)
    return _compact_anchor_refs_fallback(anchors, kind)


# =============================================================================
# Metadata lookup / validation
# =============================================================================

def _candidate_ref_keys(
    testset: Mapping[str, object], compiler: str, kind: str
) -> Sequence[str]:
    """
    Support both the prepared HumanEval generic fields and Stage-1 dataset_const
    fields such as "-O0_clang_const_refs".
    """
    opt = str(testset.get("type", ""))
    return (
        f"{kind}_refs",
        f"-{opt}_{compiler}_{kind}_refs",
        f"{opt}_{compiler}_{kind}_refs",
    )


def _find_anchor_metadata(
    testset: Mapping[str, object], compiler: str, kind: str
) -> Tuple[bool, object]:
    for key in _candidate_ref_keys(testset, compiler, kind):
        if key in testset:
            return True, testset[key]
    return False, []


def _metadata_ids(metadata: object, prefix: str) -> set:
    ids = set()

    if metadata is None:
        return ids

    if isinstance(metadata, dict):
        direct_id = metadata.get("id")
        if direct_id is not None and re.fullmatch(rf"{prefix}_\d+", str(direct_id)):
            ids.add(str(direct_id))

        for key, value in metadata.items():
            if re.fullmatch(rf"{prefix}_\d+", str(key)):
                ids.add(str(key))
            if isinstance(value, Mapping):
                ref_id = value.get("id") or value.get("token")
                if ref_id is not None and re.fullmatch(
                    rf"{prefix}_\d+", str(ref_id)
                ):
                    ids.add(str(ref_id))
        return ids

    if isinstance(metadata, list):
        for item in metadata:
            if isinstance(item, Mapping):
                ref_id = item.get("id") or item.get("token")
                if ref_id is not None and re.fullmatch(
                    rf"{prefix}_\d+", str(ref_id)
                ):
                    ids.add(str(ref_id))
            elif isinstance(item, (list, tuple)) and item:
                ref_id = str(item[0])
                if re.fullmatch(rf"{prefix}_\d+", ref_id):
                    ids.add(ref_id)

    return ids


def _anchor_metadata_matches_prompt(
    testset: Mapping[str, object], compiler: str
) -> bool:
    prepared_compiler = testset.get("anchor_compiler")
    if prepared_compiler is None:
        prepared_compiler = testset.get("string_compiler")
    if prepared_compiler is not None and str(prepared_compiler) != compiler:
        return False

    prompt = str(testset.get("input_asm_prompt", ""))
    if not prompt.strip():
        return False

    prompt_tokens = set(ANCHOR_TOKEN_RE.findall(prompt))
    metadata_tokens = set()

    for kind in ANCHOR_KINDS:
        found, metadata = _find_anchor_metadata(testset, compiler, kind)
        if found:
            metadata_tokens |= _metadata_ids(metadata, ANCHOR_PREFIX[kind])

    # A function with zero anchors is legal, but all three metadata categories
    # must be explicitly present. This accepts both generic prepared fields and
    # Stage-1 names such as -O0_clang_const_refs, while rejecting legacy
    # string-only validation cases.
    all_kinds_present = all(
        _find_anchor_metadata(testset, compiler, kind)[0]
        for kind in ANCHOR_KINDS
    )
    if not all_kinds_present:
        return False

    return prompt_tokens == metadata_tokens


# =============================================================================
# Exact C recovery
# =============================================================================

def _c_string_literal_from_bytes(raw: bytes) -> str:
    """
    Convert exact bytes to a C string literal.

    Octal escapes are fixed-width, avoiding the ambiguity of \\xNN followed by
    another hexadecimal character.
    """
    parts = ['"']
    for byte in raw:
        if byte == 0x22:
            parts.append(r"\"")
        elif byte == 0x5C:
            parts.append(r"\\")
        elif byte == 0x0A:
            parts.append(r"\n")
        elif byte == 0x0D:
            parts.append(r"\r")
        elif byte == 0x09:
            parts.append(r"\t")
        elif byte == 0x08:
            parts.append(r"\b")
        elif byte == 0x0C:
            parts.append(r"\f")
        elif byte == 0x0B:
            parts.append(r"\v")
        elif 0x20 <= byte <= 0x7E:
            parts.append(chr(byte))
        else:
            parts.append(f"\\{byte:03o}")
    parts.append('"')
    return "".join(parts)


def _decode_display_escapes(text: str) -> bytes:
    """
    Conservative fallback for old string refs without bytes_hex.
    New dataset_const refs should normally use bytes_hex instead.
    """
    out = bytearray()
    i = 0
    while i < len(text):
        ch = text[i]
        if ch != "\\":
            out.extend(ch.encode("utf-8"))
            i += 1
            continue

        if i + 1 >= len(text):
            out.append(ord("\\"))
            i += 1
            continue

        nxt = text[i + 1]
        simple = {
            "n": b"\n",
            "r": b"\r",
            "t": b"\t",
            "b": b"\b",
            "f": b"\f",
            "v": b"\v",
            "\\": b"\\",
            '"': b'"',
            "'": b"'",
            "0": b"\x00",
        }
        if nxt in simple:
            out.extend(simple[nxt])
            i += 2
            continue

        if nxt == "x" and i + 3 < len(text):
            digits = text[i + 2 : i + 4]
            if re.fullmatch(r"[0-9A-Fa-f]{2}", digits):
                out.append(int(digits, 16))
                i += 4
                continue

        out.extend(b"\\")
        out.extend(nxt.encode("utf-8"))
        i += 2

    return bytes(out)


def _string_ref_to_c_literal(ref: Mapping[str, object]) -> str:
    bytes_hex = str(ref.get("bytes_hex", "")).strip()
    if bytes_hex:
        try:
            return _c_string_literal_from_bytes(bytes.fromhex(bytes_hex))
        except ValueError:
            pass

    value = str(ref.get("value", ref.get("display", "")))
    return _c_string_literal_from_bytes(_decode_display_escapes(value))


def _format_c_float(value: object, kind: str) -> str:
    value = float(value)

    if math.isnan(value):
        return "NAN"
    if math.isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"

    if kind == "f32":
        text = format(value, ".9g")
        if not any(ch in text for ch in ".eE"):
            text += ".0"
        return text + "f"

    text = format(value, ".17g")
    if not any(ch in text for ch in ".eE"):
        text += ".0"
    return text


def _const_ref_to_c_literal(ref: Mapping[str, object]) -> str:
    c_literal = str(ref.get("c_literal", "")).strip()
    if c_literal:
        return c_literal

    kind = str(ref.get("type", ""))
    value = ref.get("value")

    if kind in {"f32", "f64"} and value is not None:
        return _format_c_float(value, kind)

    if value is None:
        raise ValueError(f"CONST ref has no recoverable value: {ref}")

    return str(value)


def _data_ref_to_c_initializer(ref: Mapping[str, object]) -> str:
    element_type = str(ref.get("type", ""))
    values = ref.get("values", [])

    if not isinstance(values, list):
        raise TypeError(f"DATA values must be a list: {ref}")

    parts = []
    for value in values:
        if element_type == "f32":
            parts.append(_format_c_float(value, "f32"))
        elif element_type == "f64":
            parts.append(_format_c_float(value, "f64"))
        else:
            parts.append(str(int(value)))

    return "{" + ", ".join(parts) + "}"


def _metadata_to_refs(metadata: object, kind: str) -> List[Mapping[str, object]]:
    """
    Normalize metadata into structured refs.

    New validation data should be a list of dictionaries. Dict token->replacement
    is accepted only as a compatibility fallback.
    """
    if metadata is None:
        return []

    if isinstance(metadata, list):
        return [item for item in metadata if isinstance(item, Mapping)]

    if isinstance(metadata, dict):
        if "id" in metadata:
            return [metadata]

        refs = []
        for token, value in metadata.items():
            if not re.fullmatch(rf"{ANCHOR_PREFIX[kind]}_\d+", str(token)):
                continue
            if isinstance(value, Mapping):
                item = dict(value)
                item.setdefault("id", str(token))
            else:
                item = {"id": str(token), "legacy_replacement": str(value)}
            refs.append(item)
        return refs

    raise TypeError(f"{kind}_refs metadata must be a list or object")


def get_recovery_map(
    testset: Mapping[str, object], compiler: str
) -> Dict[str, str]:
    recovery: Dict[str, str] = {}

    for kind in ANCHOR_KINDS:
        found, metadata = _find_anchor_metadata(testset, compiler, kind)
        if not found or metadata is None:
            continue

        for ref in _metadata_to_refs(metadata, kind):
            ref_id = str(ref.get("id", ""))
            if not ref_id:
                continue

            if "legacy_replacement" in ref:
                recovery[ref_id] = str(ref["legacy_replacement"])
            elif kind == "string":
                recovery[ref_id] = _string_ref_to_c_literal(ref)
            elif kind == "const":
                recovery[ref_id] = _const_ref_to_c_literal(ref)
            elif kind == "data":
                recovery[ref_id] = _data_ref_to_c_initializer(ref)

    return recovery


def expand_anchor_tokens(
    generated_code: str, recovery_map: Mapping[str, str]
) -> str:
    result = generated_code

    for token, replacement in sorted(
        recovery_map.items(), key=lambda item: len(item[0]), reverse=True
    ):
        result = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
            lambda _match, repl=replacement: repl,
            result,
        )

    return result


def restore_generated_code(
    generated_code: str, testset: Mapping[str, object], compiler: str
) -> str:
    return expand_anchor_tokens(
        generated_code, get_recovery_map(testset, compiler)
    )


# =============================================================================
# Build anchor-aware validation prompts
# =============================================================================

def _infer_function_name(testset: Mapping[str, object]) -> str:
    prompt = str(testset.get("input_asm_prompt", ""))
    match = re.search(r"^\s*<([^>]+)>:\s*$", prompt, flags=re.MULTILINE)
    if match:
        return match.group(1)

    explicit_name = testset.get("function_name") or testset.get("fname")
    if explicit_name:
        return str(explicit_name)

    raise ValueError(
        f"Cannot infer function name for task {testset.get('task_id', '?')} "
        "from input_asm_prompt"
    )


def _build_anchor_prompt_worker(params: Mapping[str, object]) -> Dict[str, object]:
    task_id = params.get("task_id", "?")
    opt = str(params["opt"])
    compiler = str(params["compiler"])
    function_name = str(params["function_name"])
    c_func = str(params["c_func"])
    c_prefix = str(params.get("c_prefix", ""))

    try:
        preprocessor = _load_anchor_preprocessor(
            str(params["anchor_preprocessor_path"])
        )
        _configure_sir_preprocessor_toolchain(
            preprocessor,
            clang_bin=params.get("clang_bin"),
            gcc_bin=params.get("gcc_bin"),
        )
        compiler_binary = _compiler_binary_for_preprocessor(
            preprocessor, compiler
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "source.c")
            binary_path = os.path.join(temp_dir, "source.out")

            with open(source_path, "w", encoding="utf-8") as writer:
                if c_prefix.strip():
                    writer.write(c_prefix)
                    if not c_prefix.endswith("\n"):
                        writer.write("\n")
                # ExeBench contains many small/static functions. When the SIR
                # prompt is rebuilt from source at O1-O3, clang may inline or
                # discard the target symbol, and objdump can no longer find
                # `<function_name>:`. Keep this as an evaluation-only source
                # compatibility patch: it does not change the original dataset,
                # and it is applied only to the temporary file used for anchor
                # extraction.
                escaped_name = re.escape(function_name)
                target_attr = "__attribute__((used,noinline))"
                anchored_c_func = re.sub(
                    rf"\b{escaped_name}\s*\(",
                    f"{target_attr} {function_name}(",
                    c_func,
                    count=1,
                )
                # `inline` definitions may still not emit an out-of-line body
                # in a shared object under C inline semantics. Remove it from
                # the temporary evaluation source so objdump has a concrete
                # target symbol to annotate.
                anchored_c_func = re.sub(
                    rf"\b(extern\s+)?inline\s+([^;\n]*\b{re.escape(target_attr)}\s+{escaped_name}\s*\()",
                    r"\2",
                    anchored_c_func,
                    count=1,
                )
                writer.write(anchored_c_func)

            # Match 1_get_exebench_asm_sir.py: shared PIC binary, -g3, -Oi.
            command = [
                compiler_binary,
                "-shared",
                "-fPIC",
                "-g3",
                "-fno-inline",
                "-fno-inline-functions",
                f"-{opt}",
                source_path,
                "-o",
                binary_path,
                "-lm",
            ]

            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=int(params["compile_timeout"]),
                check=False,
            )
            if completed.returncode != 0:
                error = completed.stderr.strip().splitlines()
                detail = error[-1] if error else f"exit code {completed.returncode}"
                raise RuntimeError(f"{compiler_binary} failed: {detail}")

            result = asyncio.run(
                preprocessor.decompile_bin_to_asm(
                    temp_dir,
                    binary_path,
                    function_name,
                    int(params["string_threshold"]),
                    int(params["min_string_length"]),
                    int(params["max_scan_bytes"]),
                )
            )

            # 1_get_exebench_asm_sir.py returns exactly (asm_clean, anchors).
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise TypeError(
                    "SIR decompile_bin_to_asm() must return (asm_clean, anchors)"
                )

            asm, anchors = result
            if not isinstance(asm, str):
                raise TypeError("asm_clean returned by SIR preprocessor must be a string")
            if not isinstance(anchors, list):
                raise TypeError("anchors returned by SIR preprocessor must be a list")

            string_refs = _compact_anchor_refs(preprocessor, anchors, "string")
            const_refs = _compact_anchor_refs(preprocessor, anchors, "const")
            data_refs = _compact_anchor_refs(preprocessor, anchors, "data")

        return {
            "input_asm_prompt": asm,
            "string_refs": string_refs,
            "const_refs": const_refs,
            "data_refs": data_refs,
            "compiler_binary": compiler_binary,
            "error": None,
        }

    except Exception as exc:
        return {
            "input_asm_prompt": None,
            "string_refs": None,
            "const_refs": None,
            "data_refs": None,
            "compiler_binary": None,
            "error": f"task={task_id}, opt={opt}: {exc}",
        }


def _canonicalize_sir_cross_opt_anchor_ids(
    prepared: List[dict],
    args,
) -> None:
    """Reuse the SIR generator's O0-O3 STR/CONST ID canonicalization.

    DATA_i is intentionally left object-local, matching the Stage-1 generator.
    """
    preprocessor = _load_anchor_preprocessor(
        getattr(args, "anchor_preprocessor_path", str(DEFAULT_ANCHOR_PREPROCESSOR))
    )
    _configure_sir_preprocessor_toolchain(
        preprocessor,
        clang_bin=getattr(args, "clang_bin", None),
        gcc_bin=getattr(args, "gcc_bin", None),
    )

    canonicalize = getattr(
        preprocessor, "_canonicalize_cross_opt_scalar_anchor_ids", None
    )
    if not callable(canonicalize):
        logger.warning(
            "SIR preprocessor has no _canonicalize_cross_opt_scalar_anchor_ids(); "
            "keeping per-optimization anchor IDs unchanged"
        )
        return

    groups: Dict[str, List[int]] = {}
    for index, testset in enumerate(prepared):
        try:
            group_key = _infer_function_name(testset)
        except Exception:
            group_key = str(testset.get("task_id", index))
        groups.setdefault(group_key, []).append(index)

    for indices in groups.values():
        asm_codes: Dict[str, object] = {}
        index_by_opt: Dict[str, int] = {}

        for index in indices:
            testset = prepared[index]
            opt = str(testset.get("type", ""))
            if opt not in OPT_PROMPTS:
                continue
            index_by_opt[opt] = index
            prefix = f"-{opt}_{args.compiler}"
            asm_codes[prefix] = str(testset.get("input_asm_prompt", ""))
            asm_codes[f"{prefix}_string_refs"] = list(
                _metadata_to_refs(
                    _find_anchor_metadata(testset, args.compiler, "string")[1],
                    "string",
                )
            )
            asm_codes[f"{prefix}_const_refs"] = list(
                _metadata_to_refs(
                    _find_anchor_metadata(testset, args.compiler, "const")[1],
                    "const",
                )
            )
            asm_codes[f"{prefix}_data_refs"] = list(
                _metadata_to_refs(
                    _find_anchor_metadata(testset, args.compiler, "data")[1],
                    "data",
                )
            )

        canonicalize(asm_codes)

        for opt, index in index_by_opt.items():
            prefix = f"-{opt}_{args.compiler}"
            if isinstance(asm_codes.get(prefix), str):
                prepared[index]["input_asm_prompt"] = asm_codes[prefix]
            prepared[index]["string_refs"] = asm_codes.get(
                f"{prefix}_string_refs", []
            )
            prepared[index]["const_refs"] = asm_codes.get(
                f"{prefix}_const_refs", []
            )
            prepared[index]["data_refs"] = asm_codes.get(
                f"{prefix}_data_refs", []
            )


def _mark_sir_prepared_case(testset: dict, args, compiler_binary: Optional[str] = None) -> None:
    testset["anchor_compiler"] = args.compiler
    testset["string_compiler"] = args.compiler
    testset["anchor_schema_version"] = PREPARED_SCHEMA_VERSION
    testset["anchor_preprocessor"] = os.path.basename(
        str(getattr(args, "anchor_preprocessor_path", DEFAULT_ANCHOR_PREPROCESSOR))
    )
    if compiler_binary:
        testset["anchor_compiler_binary"] = str(compiler_binary)


def _write_json(path: str, data: object) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as writer:
        json.dump(data, writer, indent=2, ensure_ascii=False)


def _load_prepared_cache(
    path: Optional[str],
    expected_count: int,
    compiler: str,
    expected_preprocessor_path: str,
) -> Optional[List[dict]]:
    if not path or not os.path.isfile(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as reader:
            prepared = json.load(reader)
    except Exception as exc:
        logger.warning(f"Failed to load prepared test set {path}: {exc}")
        return None

    if not isinstance(prepared, list) or len(prepared) != expected_count:
        count = len(prepared) if isinstance(prepared, list) else "not-a-list"
        logger.warning(
            f"Ignoring prepared test set with wrong case count: "
            f"{count} != {expected_count}"
        )
        return None

    expected_name = os.path.basename(str(expected_preprocessor_path))
    schema_invalid = [
        index
        for index, testset in enumerate(prepared)
        if testset.get("anchor_schema_version") != PREPARED_SCHEMA_VERSION
        or os.path.basename(str(testset.get("anchor_preprocessor", ""))) != expected_name
    ]
    if schema_invalid:
        logger.warning(
            f"Ignoring non-SIR/stale prepared cache: {len(schema_invalid)} case(s) "
            "were not built with the current SIR evaluation schema"
        )
        return None

    invalid = [
        index
        for index, testset in enumerate(prepared)
        if not _anchor_metadata_matches_prompt(testset, compiler)
    ]
    if invalid:
        logger.warning(
            f"Ignoring stale/inconsistent prepared test set: "
            f"{len(invalid)} case(s) failed STR/CONST/DATA validation"
        )
        return None

    logger.info(f"Loading cached SIR-compatible STR/CONST/DATA test set: {path}")
    return prepared


def prepare_anchor_testsets(testsets: List[dict], args) -> List[dict]:
    mode = args.string_prompt_mode

    if mode == "auto":
        cached = _load_prepared_cache(
            getattr(args, "prepared_testset_output", None),
            len(testsets),
            args.compiler,
            str(getattr(args, "anchor_preprocessor_path", DEFAULT_ANCHOR_PREPROCESSOR)),
        )
        if cached is not None:
            return cached

    prepared = [dict(testset) for testset in testsets]
    rebuild_indices = []

    expected_preprocessor_name = os.path.basename(
        str(getattr(args, "anchor_preprocessor_path", DEFAULT_ANCHOR_PREPROCESSOR))
    )

    for index, testset in enumerate(prepared):
        metadata_matches = _anchor_metadata_matches_prompt(testset, args.compiler)
        sir_schema_matches = (
            testset.get("anchor_schema_version") == PREPARED_SCHEMA_VERSION
            and os.path.basename(str(testset.get("anchor_preprocessor", "")))
            == expected_preprocessor_name
        )

        # In auto mode, never silently reuse a dataset_const/legacy prepared
        # prompt merely because its token set happens to match. Rebuild unless
        # it carries this evaluator's SIR schema marker.
        if mode == "rebuild" or (
            mode == "auto" and not (metadata_matches and sir_schema_matches)
        ):
            rebuild_indices.append(index)
        elif mode == "existing" and not metadata_matches:
            raise ValueError(
                "The test set is not STR/CONST/DATA-aware, or its prompt and "
                f"metadata disagree for task {testset.get('task_id', '?')} "
                f"({testset.get('type', '?')}). Use --string_prompt_mode "
                "auto/rebuild or provide a prepared anchor-aware test set."
            )

    if rebuild_indices:
        logger.info(
            f"Rebuilding {len(rebuild_indices)} SIR-compatible STR/CONST/DATA-aware "
            f"assembly prompts with {args.compiler}"
        )

        jobs = []
        for index in rebuild_indices:
            testset = prepared[index]
            jobs.append(
                {
                    "task_id": testset.get("task_id", index),
                    "opt": testset["type"],
                    "compiler": args.compiler,
                    "function_name": _infer_function_name(testset),
                    "c_func": testset["c_func"],
                    "c_prefix": testset.get("c_prefix", ""),
                    "string_threshold": args.string_threshold,
                    "min_string_length": args.min_string_length,
                    "max_scan_bytes": args.max_scan_bytes,
                    "compile_timeout": args.compile_timeout,
                    "clang_bin": getattr(args, "clang_bin", None),
                    "gcc_bin": getattr(args, "gcc_bin", None),
                    "anchor_preprocessor_path": getattr(
                        args,
                        "anchor_preprocessor_path",
                        str(DEFAULT_ANCHOR_PREPROCESSOR),
                    ),
                }
            )

        if args.prompt_workers <= 1:
            results = [
                _build_anchor_prompt_worker(job)
                for job in tqdm(jobs, desc="Preparing SIR-compatible ASM prompts")
            ]
        else:
            with multiprocessing.Pool(args.prompt_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(_build_anchor_prompt_worker, jobs),
                        total=len(jobs),
                        desc="Preparing SIR-compatible ASM prompts",
                    )
                )

        errors = [str(result["error"]) for result in results if result["error"]]
        if errors:
            preview = "\n".join(errors[:10])
            remainder = len(errors) - min(10, len(errors))
            if remainder:
                preview += f"\n... and {remainder} more failures"
            raise RuntimeError(
                "Failed to prepare STR/CONST/DATA-aware validation cases:\n"
                + preview
            )

        for index, result in zip(rebuild_indices, results):
            prepared[index]["input_asm_prompt"] = result["input_asm_prompt"]
            prepared[index]["string_refs"] = result["string_refs"]
            prepared[index]["const_refs"] = result["const_refs"]
            prepared[index]["data_refs"] = result["data_refs"]
            _mark_sir_prepared_case(
                prepared[index],
                args,
                compiler_binary=result.get("compiler_binary"),
            )

    # SIR Stage 1 canonicalizes STR/CONST IDs jointly across O0-O3. Reuse the
    # exact same function here so HumanEval prompts follow the training scheme.
    _canonicalize_sir_cross_opt_anchor_ids(prepared, args)
    for item in prepared:
        _mark_sir_prepared_case(item, args, item.get("anchor_compiler_binary"))

    invalid = [
        index
        for index, testset in enumerate(prepared)
        if not _anchor_metadata_matches_prompt(testset, args.compiler)
    ]
    if invalid:
        raise RuntimeError(
            f"Prepared validation set is inconsistent for {len(invalid)} case(s). "
            f"First invalid indices: {invalid[:10]}"
        )

    if getattr(args, "prepared_testset_output", None):
        _write_json(args.prepared_testset_output, prepared)
        logger.info(f"Saved prepared test set to {args.prepared_testset_output}")

    str_count = sum(
        len(_metadata_to_refs(_find_anchor_metadata(item, args.compiler, "string")[1], "string"))
        for item in prepared
    )
    const_count = sum(
        len(_metadata_to_refs(_find_anchor_metadata(item, args.compiler, "const")[1], "const"))
        for item in prepared
    )
    data_count = sum(
        len(_metadata_to_refs(_find_anchor_metadata(item, args.compiler, "data")[1], "data"))
        for item in prepared
    )
    logger.info(
        f"Prepared {len(prepared)} cases with STR={str_count}, "
        f"CONST={const_count}, DATA={data_count}, "
        f"total anchors={str_count + const_count + data_count}"
    )

    return prepared


# Backward-compatible alias. Old batch code can still call this name if needed.
prepare_string_testsets = prepare_anchor_testsets


# =============================================================================
# vLLM / evaluation
# =============================================================================

def cleanup_vllm(llm) -> None:
    try:
        engine = getattr(llm, "llm_engine", None)
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception as exc:
        logger.warning(f"Failed to shut down vLLM cleanly: {exc}")

    del llm
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        logger.warning(f"Failed to clear the CUDA cache: {exc}")

    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        try:
            from vllm.model_executor.parallel_utils.parallel_state import (
                destroy_model_parallel,
            )

            destroy_model_parallel()
        except Exception:
            pass


def load_testsets(testset_path: str) -> List[dict]:
    with open(testset_path, "r", encoding="utf-8") as reader:
        testsets = json.load(reader)

    if not isinstance(testsets, list):
        raise TypeError(f"Test set must be a JSON array: {testset_path}")

    logger.info(f"Loaded test set with {len(testsets)} cases")
    return testsets


def build_inputs(testsets: Sequence[Mapping[str, object]]) -> List[str]:
    inputs = []
    for testset in testsets:
        opt = str(testset["type"])
        if opt not in OPT_PROMPTS:
            raise ValueError(f"Unsupported optimization type: {opt}")

        input_asm_prompt = str(testset["input_asm_prompt"]).strip()
        inputs.append(OPT_PROMPTS[opt] + input_asm_prompt + AFTER_PROMPT)

    return inputs


def prepare_eval_context(args) -> Dict[str, object]:
    from transformers import AutoTokenizer

    tokenizer_path = getattr(args, "tokenizer_path", None) or getattr(
        args, "model_path", None
    )
    if not tokenizer_path:
        raise ValueError("tokenizer_path or model_path is required")

    testsets = load_testsets(args.testset_path)
    testsets = prepare_anchor_testsets(testsets, args)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    stop_sequences = [tokenizer.eos_token] if tokenizer.eos_token else None

    return {
        "testsets": testsets,
        "inputs": build_inputs(testsets),
        "stop_sequences": stop_sequences,
        "opts": tuple(OPT_PROMPTS),
    }


def build_llm(args):
    from vllm import LLM

    return LLM(
        model=args.model_path,
        tokenizer=args.tokenizer_path or args.model_path,
        tensor_parallel_size=args.gpus,
        max_model_len=args.max_total_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def generate_results(llm, inputs: Sequence[str], stop_sequences, args):
    from vllm import SamplingParams
    from transformers import AutoTokenizer

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop=stop_sequences,
    )

    tokenizer_path = getattr(args, "tokenizer_path", None) or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prompt_lengths = [
        len(tokenizer.encode(text, add_special_tokens=True)) for text in inputs
    ]
    max_prompt_tokens = max(1, args.max_total_tokens - args.max_new_tokens)
    eligible_indices = [
        index
        for index, length in enumerate(prompt_lengths)
        if length <= max_prompt_tokens
    ]
    overflow_count = len(inputs) - len(eligible_indices)
    if overflow_count:
        logger.warning(
            f"Skipping {overflow_count} prompt(s) longer than "
            f"{max_prompt_tokens} tokens before generation"
        )

    repeated_results = []
    logger.info(f"Generation will run {args.repeat} time(s)")

    for repeat_index in range(args.repeat):
        logger.info(f"Generation repeat {repeat_index + 1}/{args.repeat}")
        outputs = [[""] for _ in inputs]
        if eligible_indices:
            results = llm.generate(
                [inputs[index] for index in eligible_indices],
                sampling_params,
            )
            for index, result in zip(eligible_indices, results):
                outputs[index] = [result.outputs[0].text]
        repeated_results.append(outputs)

    return repeated_results


def save_generation_outputs(
    testsets: Sequence[Mapping[str, object]],
    gen_results,
    output_path: Optional[str],
    compiler: str,
) -> None:
    if not output_path:
        return

    saved_data = []
    for testset, result in zip(testsets, gen_results):
        item = dict(testset)
        item["output"] = result[0]
        item["output_restored"] = restore_generated_code(
            result[0], testset, compiler
        )
        saved_data.append(item)

    _write_json(output_path, saved_data)


def load_generation_outputs(output_path: Optional[str], expected_count: int):
    if not output_path or not os.path.exists(output_path):
        return None

    try:
        with open(output_path, "r", encoding="utf-8") as reader:
            saved_data = json.load(reader)
    except Exception as exc:
        logger.warning(
            f"Failed to load existing generation output {output_path}: {exc}"
        )
        return None

    if not isinstance(saved_data, list) or len(saved_data) != expected_count:
        saved_count = len(saved_data) if isinstance(saved_data, list) else "not a list"
        logger.warning(
            f"Existing generation output count mismatch: "
            f"{saved_count} != {expected_count}"
        )
        return None

    if any("output" not in item for item in saved_data):
        logger.warning(
            f"Existing generation output is missing an output field: {output_path}"
        )
        return None

    logger.info(f"Reusing existing generation output: {output_path}")
    return [[[item["output"]] for item in saved_data]]


def _split_includes(source: str) -> Tuple[str, str]:
    includes = []
    body = []

    for line in source.splitlines():
        if line.lstrip().startswith("#include"):
            includes.append(line)
        else:
            body.append(line)

    return "\n".join(includes), "\n".join(body)


def evaluate_func(params: Mapping[str, object]) -> Tuple[int, int]:
    c_func = str(params["c_func"])
    c_prefix = str(params.get("c_prefix", ""))
    c_test = str(params["c_test"])
    generated_code = str(params["c_func_decompile"])
    recovery_map = params.get("recovery_map", {})

    if not isinstance(recovery_map, dict):
        raise TypeError("recovery_map must be an object")

    c_func_decompile = expand_anchor_tokens(generated_code, recovery_map)

    func_includes, _ = _split_includes(c_prefix + "\n" + c_func)
    test_includes, test_body = _split_includes(c_test)
    includes = "\n".join(
        part for part in (func_includes, test_includes) if part
    )

    prefix_includes, prefix_body = _split_includes(c_prefix)
    _ = prefix_includes
    c_combine = includes + "\n\n" + prefix_body + "\n\n" + c_func_decompile + "\n\n" + test_body
    c_onlyfunc = includes + "\n\n" + prefix_body + "\n\n" + c_func_decompile

    compile_timeout = int(params.get("compile_timeout", 10))
    run_timeout = int(params.get("run_timeout", 10))

    with tempfile.TemporaryDirectory() as temp_dir:
        c_file = os.path.join(temp_dir, "combine.c")
        executable = os.path.join(temp_dir, "combine")
        c_file_onlyfunc = os.path.join(temp_dir, "onlyfunc.c")
        assembly_file = os.path.join(temp_dir, "onlyfunc.s")

        with open(c_file, "w", encoding="utf-8") as writer:
            writer.write(c_combine)
        with open(c_file_onlyfunc, "w", encoding="utf-8") as writer:
            writer.write(c_onlyfunc)

        try:
            subprocess.run(
                ["gcc", "-S", c_file_onlyfunc, "-o", assembly_file, "-lm"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=compile_timeout,
            )
        except (OSError, subprocess.SubprocessError):
            return 0, 0

        try:
            subprocess.run(
                ["gcc", c_file, "-o", executable, "-lm"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=compile_timeout,
            )
        except (OSError, subprocess.SubprocessError):
            return 1, 0

        try:
            subprocess.run(
                [executable],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=run_timeout,
            )
        except (OSError, subprocess.SubprocessError):
            return 1, 0

    return 1, 1


def decompile_pass_rate(testsets, gen_results_repeat, opts, args) -> int:
    if not gen_results_repeat:
        raise ValueError("No generation results to evaluate")

    all_stats = []

    for repeat_index, gen_results in enumerate(gen_results_repeat):
        if len(gen_results) != len(testsets):
            raise ValueError(
                f"Generation count mismatch in repeat {repeat_index}: "
                f"{len(gen_results)} != {len(testsets)}"
            )

        tasks = [
            {
                "c_func": testset["c_func"],
                "c_prefix": testset.get("c_prefix", ""),
                "c_test": testset["c_test"],
                "c_func_decompile": output[0],
                "recovery_map": get_recovery_map(testset, args.compiler),
                "compile_timeout": args.compile_timeout,
                "run_timeout": args.run_timeout,
            }
            for testset, output in zip(testsets, gen_results)
        ]

        if args.num_workers <= 1:
            eval_results = [
                evaluate_func(task)
                for task in tqdm(tasks, desc="Compiling")
            ]
        else:
            with multiprocessing.Pool(args.num_workers) as pool:
                eval_results = list(
                    tqdm(
                        pool.imap(evaluate_func, tasks),
                        total=len(tasks),
                        desc="Compiling",
                    )
                )

        stats = {
            opt: {"compile": 0, "run": 0, "total": 0}
            for opt in opts
        }

        for testset, (flag_compile, flag_run) in zip(testsets, eval_results):
            opt = str(testset["type"])
            stats[opt]["total"] += 1
            stats[opt]["compile"] += flag_compile
            stats[opt]["run"] += flag_run

        all_stats.append(stats)

    results = {}

    for opt in opts:
        compile_count = sum(stats[opt]["compile"] for stats in all_stats)
        run_count = sum(stats[opt]["run"] for stats in all_stats)
        total_count = sum(stats[opt]["total"] for stats in all_stats)

        compile_rate = compile_count / total_count if total_count else 0
        run_rate = run_count / total_count if total_count else 0

        results[opt] = {
            "compile_rate": round(compile_rate, 4),
            "run_rate": round(run_rate, 4),
        }

        logger.info(
            f"Optimization {opt}: "
            f"compile rate={compile_rate:.4f}, run rate={run_rate:.4f}"
        )

    num_opts = len(opts)
    results["mean"] = {
        "compile_rate": round(
            sum(results[opt]["compile_rate"] for opt in opts)
            * 100
            / num_opts,
            2,
        ),
        "run_rate": round(
            sum(results[opt]["run_rate"] for opt in opts)
            * 100
            / num_opts,
            2,
        ),
    }

    if args.output_result_path:
        _write_json(args.output_result_path, results)

    return 0


def run_eval_pipeline(args) -> int:
    return run_eval_pipeline_with_context(args, context=None)


def run_eval_pipeline_with_context(args, context=None) -> int:
    try:
        if args.repeat < 1:
            raise ValueError("--repeat must be at least 1")

        if context is None:
            context = prepare_eval_context(args)

        testsets = context["testsets"]
        inputs = context["inputs"]
        stop_sequences = context["stop_sequences"]
        opts = context["opts"]

        gen_results_repeat = None
        reused_outputs = False

        if getattr(args, "reuse_outputs", False) and args.repeat == 1:
            gen_results_repeat = load_generation_outputs(
                args.output_path, len(testsets)
            )
            reused_outputs = gen_results_repeat is not None

        if gen_results_repeat is None:
            llm = None
            try:
                logger.info(f"Loading model: {args.model_path}")
                llm = build_llm(args)
                gen_results_repeat = generate_results(
                    llm, inputs, stop_sequences, args
                )
            finally:
                if llm is not None:
                    logger.info("Releasing vLLM model resources")
                    loaded_llm = llm
                    llm = None
                    cleanup_vllm(loaded_llm)

        if not reused_outputs:
            save_generation_outputs(
                testsets,
                gen_results_repeat[0],
                args.output_path,
                args.compiler,
            )

        return decompile_pass_rate(
            testsets,
            gen_results_repeat,
            opts,
            args,
        )

    except Exception as exc:
        logger.error(exc)
        traceback.print_exc()
        return -1


def main() -> None:
    args = parse_args()
    sys.exit(run_eval_pipeline(args))


if __name__ == "__main__":
    main()
