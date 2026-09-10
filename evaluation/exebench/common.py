#!/usr/bin/env python3
"""Shared utilities for the executable ExeBench protocol."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import zstandard


PROJECT = Path(__file__).resolve().parents[2]
OPTIMIZATIONS = ("O0", "O1", "O2", "O3")
DEFAULT_DATASET_ROOT = Path("/home/jiang/share/dataset/exebench")
DEFAULT_DATASET = PROJECT / "decompile-eval/exebench-test-real-c-o0-o3.json"
DEFAULT_METADATA = (
    PROJECT / "decompile-eval/exebench-test-real-c-metadata.json"
)
DEFAULT_REFERENCE_VALIDATION = (
    PROJECT / "results/metrics/exebench-reference-validation.json"
)
DEFAULT_RUNTIME = (
    PROJECT.parent
    / "third_party/slade_artifact/slade_module/exebench"
)

FENCE_RE = re.compile(r"```(?:c|C)?\s*(.*?)```", re.DOTALL)
INLINE_RE = re.compile(
    r"\b(?:static|inline|__inline|__inline__)\b(?:\s+__attribute__\s*"
    r"\(\(\s*always_inline\s*\)\))?"
)
WRAPPER_INCLUDE_RE = re.compile(
    r'extern\s*"C"\s*\{\s*#include\s+"[^"]+"\s*\}', re.DOTALL
)


@dataclass
class ExeResult:
    compilable: bool = False
    executable: bool = False
    all_tests_pass: bool = False
    tests_passed: int = 0
    tests_total: int = 0
    compile_stage: str = "source"
    compile_returncode: int | None = None
    first_failed_test: int | None = None
    timed_out: bool = False
    stderr: str = ""
    accepted_function_name: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def load_test_real(dataset_root: Path) -> list[dict]:
    """Read the official compressed JSONL split without downloading anything."""

    archive = dataset_root / "test_real.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(f"ExeBench test archive not found: {archive}")
    rows: list[dict] = []
    with tarfile.open(archive, "r:gz") as tar:
        members = sorted(
            (
                member
                for member in tar.getmembers()
                if member.isfile() and member.name.endswith(".jsonl.zst")
            ),
            key=lambda member: member.name,
        )
        if not members:
            raise ValueError(f"{archive} contains no .jsonl.zst members")
        decompressor = zstandard.ZstdDecompressor()
        for member in members:
            compressed = tar.extractfile(member)
            if compressed is None:
                continue
            with decompressor.stream_reader(compressed) as reader:
                pending = b""
                while True:
                    chunk = reader.read(1024 * 1024)
                    if not chunk:
                        break
                    pending += chunk
                    lines = pending.split(b"\n")
                    pending = lines.pop()
                    for line in lines:
                        if line.strip():
                            record = json.loads(line)
                            rows.append(record.get("text", record))
                if pending.strip():
                    record = json.loads(pending)
                    rows.append(record.get("text", record))
    return rows


def extract_c_code(text: str) -> str:
    matches = FENCE_RE.findall(text or "")
    if matches:
        return max(matches, key=len).strip()
    return (text or "").strip()


def normalize_function_source(source: str) -> str:
    """Make the target definition visible without changing local storage."""

    header, separator, body = source.partition("{")
    if not separator:
        return INLINE_RE.sub("", source)
    return INLINE_RE.sub("", header) + separator + body


def adapt_candidate_name(source: str, expected: str) -> tuple[str, str | None]:
    """Accept Nova/benchmark-style ``func0`` as a non-semantic symbol alias."""

    expected_definition = re.compile(
        rf"\b{re.escape(expected)}\s*\([^;{{}}]*\)\s*(?:__attribute__"
        rf"\s*\(\([^)]*\)\)\s*)?\{{",
        re.DOTALL,
    )
    if expected_definition.search(source):
        return source, expected
    func0_definition = re.compile(
        r"\bfunc0\s*(?=\([^;{}]*\)\s*(?:__attribute__\s*"
        r"\(\([^)]*\)\)\s*)?\{)",
        re.DOTALL,
    )
    adapted, count = func0_definition.subn(expected, source, count=1)
    return (adapted, "func0") if count else (source, None)


def compile_function_gas(
    source: str,
    function_name: str,
    optimization: str,
    *,
    gcc: str = "gcc",
    timeout: float = 30.0,
) -> str:
    """Compile and extract the target GAS body as used by SLaDe."""

    if optimization not in OPTIMIZATIONS:
        raise ValueError(f"unsupported optimization: {optimization}")
    command = [gcc, "-S", f"-{optimization}", "-x", "c"]
    if Path(gcc).name.startswith("clang"):
        # ExeBench contains legacy GNU C with implicit declarations. Match
        # GCC's permissive front-end behavior while using Clang for codegen.
        command.extend(
            [
                "-Wno-implicit-function-declaration",
                "-Wno-int-conversion",
            ]
        )
    command.extend(["-o", "-", "-"])
    process = subprocess.run(
        command,
        input=source,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr[-4000:])

    function: list[str] = [
        f".globl {function_name}",
        f".type {function_name}, @function",
    ]
    inside = False
    complete = False
    label = re.compile(rf"^\s*{re.escape(function_name)}:")
    for line in process.stdout.splitlines():
        if label.match(line):
            inside = True
        if inside:
            without_comment = line.split("#", 1)[0]
            if without_comment.split():
                function.append(without_comment)
        if inside and ".cfi_endproc" in line:
            complete = True
            break
    if not inside or not complete:
        raise RuntimeError(
            f"could not extract {function_name} GAS assembly at {optimization}"
        )
    return "\n".join(function) + "\n"


def io_dict(items: list[dict] | dict) -> dict:
    if isinstance(items, dict):
        return items
    result = {}
    for item in items:
        value = item["value"]
        try:
            result[item["var"]] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            result[item["var"]] = ast.literal_eval(value)
    return result


def diff_io(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        # JSON has no tuple distinction; tolerate tuple/list dataset artifacts.
        if isinstance(observed, (list, tuple)) and isinstance(
            expected, (list, tuple)
        ):
            observed, expected = list(observed), list(expected)
        else:
            return False
    if isinstance(observed, list):
        return len(observed) == len(expected) and all(
            diff_io(left, right)
            for left, right in zip(observed, expected)
        )
    if isinstance(observed, dict):
        return observed.keys() == expected.keys() and all(
            diff_io(observed[key], expected[key]) for key in observed
        )
    if isinstance(observed, float):
        return math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-12)
    return observed == expected


def _defined_symbols(object_path: Path, *, global_only: bool) -> list[str]:
    command = ["nm"]
    if global_only:
        command.append("-g")
    command.extend(["--defined-only", str(object_path)])
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    )
    symbols = []
    for line in process.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3:
            symbols.append(fields[-1])
    return symbols


def _run(
    command: list[str],
    *,
    timeout: float,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        cwd=cwd,
        start_new_session=True,
    )


def evaluate_source(
    row: dict,
    prediction: str,
    *,
    gcc: str = "gcc",
    gxx: str = "g++",
    objcopy: str = "objcopy",
    runtime: Path = DEFAULT_RUNTIME,
    compile_timeout: float = 30.0,
    test_timeout: float = 5.0,
) -> ExeResult:
    """Compile one candidate against ExeBench's wrapper and run all real I/O."""

    pairs = row["real_io_pairs"]
    result = ExeResult(tests_total=len(pairs))
    candidate = normalize_function_source(extract_c_code(prediction))
    if not candidate:
        result.compile_stage = "missing_source"
        result.stderr = "prediction contains no C source"
        return result
    function_name = row["function_name"]
    candidate, accepted_name = adapt_candidate_name(candidate, function_name)
    result.accepted_function_name = accepted_name
    if accepted_name is None:
        result.compile_stage = "missing_function"
        result.stderr = (
            f"candidate defines neither {function_name} nor canonical func0"
        )
        return result
    if not (runtime / "nlohmann/json.hpp").is_file():
        result.compile_stage = "runtime"
        result.stderr = f"missing ExeBench runtime: {runtime}"
        return result

    dependencies = row.get("c_prefix", row.get("real_deps", ""))
    signature = row["func_head_types"].replace("extern", "").strip()
    with tempfile.TemporaryDirectory(prefix="decir-exebench-") as tmp:
        root = Path(tmp)
        source_path = root / "candidate.c"
        object_path = root / "candidate.o"
        localized_path = root / "candidate-local.o"
        deps_path = root / "deps.c"
        wrapper_path = root / "wrapper.cpp"
        executable = root / "candidate.x"
        source_path.write_text(
            dependencies + "\n" + candidate + "\n", encoding="utf-8"
        )
        deps_path.write_text(
            dependencies + f"\nextern {signature};\n", encoding="utf-8"
        )
        wrapper = WRAPPER_INCLUDE_RE.sub(
            'extern "C" {\n#include "' + str(deps_path) + '"\n}',
            row["real_exe_wrapper"],
            count=1,
        )
        if wrapper == row["real_exe_wrapper"]:
            result.compile_stage = "wrapper_rewrite"
            result.stderr = "could not replace ExeBench temporary include"
            return result
        wrapper_path.write_text(wrapper, encoding="utf-8")

        try:
            source_command = [
                gcc,
                "-c",
                "-O0",
                "-fPIC",
                "-ffunction-sections",
                "-fdata-sections",
                "-Wno-implicit-function-declaration",
            ]
            if Path(gcc).name.startswith("clang"):
                # Match GCC's permissive handling of legacy benchmark C while
                # retaining Clang as the actual compiler and code generator.
                source_command.append("-Wno-int-conversion")
            source_command.extend(
                [str(source_path), "-o", str(object_path)]
            )
            process = _run(source_command, timeout=compile_timeout)
            result.compile_returncode = process.returncode
            result.stderr = process.stderr[-4000:]
            if process.returncode != 0:
                return result
            symbols = _defined_symbols(object_path, global_only=False)
            if function_name not in symbols:
                result.compile_stage = "missing_symbol"
                result.stderr = (
                    result.stderr + f"\nmissing object symbol {function_name}"
                )[-4000:]
                return result
            global_symbols = _defined_symbols(object_path, global_only=True)
            localize = [
                symbol for symbol in global_symbols if symbol != function_name
            ]
            command = [objcopy, f"--globalize-symbol={function_name}"]
            for symbol in localize:
                command.append(f"--localize-symbol={symbol}")
            command.extend([str(object_path), str(localized_path)])
            process = _run(command, timeout=compile_timeout)
            if process.returncode != 0:
                result.compile_stage = "symbol_isolation"
                result.compile_returncode = process.returncode
                result.stderr = (result.stderr + "\n" + process.stderr)[-4000:]
                return result
            result.compile_stage = "wrapper"
            wrapper_command = [gxx]
            # GCC uses -fpermissive for compatibility with the upstream
            # wrapper, while Clang does not implement that option.
            if not Path(gxx).name.startswith("clang"):
                wrapper_command.append("-fpermissive")
            wrapper_command.extend(
                [
                    "-O0",
                    "-Wl,--gc-sections",
                    str(wrapper_path),
                    str(localized_path),
                    "-I",
                    str(runtime),
                    "-lm",
                    "-ldl",
                    "-pthread",
                    "-o",
                    str(executable),
                ]
            )
            process = _run(wrapper_command, timeout=compile_timeout)
            result.compile_returncode = process.returncode
            result.stderr = (result.stderr + "\n" + process.stderr)[-4000:]
            if process.returncode != 0 or not executable.is_file():
                return result
        except subprocess.TimeoutExpired as exc:
            result.timed_out = True
            result.stderr = (result.stderr + "\n" + str(exc))[-4000:]
            return result
        except (OSError, subprocess.CalledProcessError) as exc:
            result.stderr = (result.stderr + "\n" + str(exc))[-4000:]
            return result

        result.compilable = True
        result.executable = True
        result.compile_stage = "complete"
        for index, pair in enumerate(pairs):
            input_path = root / f"input-{index}.json"
            output_path = root / f"output-{index}.json"
            input_path.write_text(
                json.dumps(io_dict(pair["input"])), encoding="utf-8"
            )
            try:
                process = _run(
                    [str(executable), str(input_path), str(output_path)],
                    timeout=test_timeout,
                    cwd=root,
                )
                if process.returncode == 0 and output_path.is_file():
                    observed = json.loads(output_path.read_text(encoding="utf-8"))
                    if diff_io(observed, io_dict(pair["output"])):
                        result.tests_passed += 1
                        continue
                if result.first_failed_test is None:
                    result.first_failed_test = index
                result.stderr = (
                    result.stderr + "\n" + process.stderr
                )[-4000:]
            except subprocess.TimeoutExpired as exc:
                result.timed_out = True
                if result.first_failed_test is None:
                    result.first_failed_test = index
                result.stderr = (result.stderr + "\n" + str(exc))[-4000:]
            except (OSError, json.JSONDecodeError) as exc:
                if result.first_failed_test is None:
                    result.first_failed_test = index
                result.stderr = (result.stderr + "\n" + str(exc))[-4000:]
        result.all_tests_pass = result.tests_passed == result.tests_total
    return result
