"""Shared compilation and fixed-test evaluation utilities."""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


INCLUDE_RE = re.compile(r"^\s*#\s*include\b.*$", re.MULTILINE)
FENCE_RE = re.compile(r"```(?:c|C)?\s*(.*?)```", re.DOTALL)


@dataclass
class EvaluationResult:
    compilable: bool = False
    re_executable: bool = False
    compile_stage: str = "source"
    compile_returncode: int | None = None
    run_returncode: int | None = None
    timed_out: bool = False
    stderr: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def extract_c_code(text: str) -> str:
    matches = FENCE_RE.findall(text)
    if matches:
        return max(matches, key=len).strip()
    return text.strip()


def split_includes(*parts: str) -> tuple[str, list[str]]:
    includes: list[str] = []
    cleaned: list[str] = []
    for part in parts:
        for include in INCLUDE_RE.findall(part):
            if include not in includes:
                includes.append(include)
        cleaned.append(INCLUDE_RE.sub("", part))
    return "\n".join(includes) + ("\n" if includes else ""), cleaned


def compile_and_test(
    reference: str,
    tests: str,
    prediction: str,
    *,
    prefix: str = "",
    compiler: str = "gcc",
    timeout: float = 10.0,
    strict: bool = True,
    run_tests: bool = True,
) -> EvaluationResult:
    """Compile a candidate and optionally execute the supplied test harness."""

    prediction = extract_c_code(prediction)
    if not prediction:
        return EvaluationResult(
            compile_stage="missing_source",
            stderr="prediction contains no C source",
        )
    includes, (reference, tests, prediction, prefix) = split_includes(
        reference, tests, prediction, prefix
    )
    del reference  # only its includes are needed by the original protocol
    # ``strict=False`` intentionally reproduces the original
    # HumanEval-Decompile command line: no explicit language dialect or warning
    # policy, so GCC uses its configured GNU C default.  Keeping even seemingly
    # innocuous flags such as ``-std=c11`` here changes several benchmark
    # outcomes and is therefore not a faithful legacy comparison.
    flags: list[str] = []
    if strict:
        flags = [
            "-std=c11",
            "-fno-strict-aliasing",
            "-Wall",
            "-Wextra",
            "-Werror=implicit-function-declaration",
            "-Werror=incompatible-pointer-types",
        ]

    result = EvaluationResult()
    with tempfile.TemporaryDirectory(prefix="decir-eval-") as tmp:
        tmpdir = Path(tmp)
        source_only = tmpdir / "candidate.c"
        combined = tmpdir / "combined.c"
        object_file = tmpdir / "candidate.o"
        executable = tmpdir / "combined"
        source_only.write_text(
            includes + "\n" + prefix + "\n" + prediction + "\n", encoding="utf-8"
        )
        combined.write_text(
            includes + "\n" + prefix + "\n" + prediction + "\n" + tests + "\n",
            encoding="utf-8",
        )

        source_command = [
            compiler,
            *flags,
            "-c",
            str(source_only),
            "-o",
            str(object_file),
        ]
        try:
            proc = subprocess.run(
                source_command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            result.timed_out = True
            result.stderr = str(exc)
            return result
        result.compile_returncode = proc.returncode
        result.stderr = proc.stderr[-4000:]
        if proc.returncode != 0:
            return result
        result.compilable = True

        if not run_tests or not tests.strip():
            result.compile_stage = "complete"
            return result

        result.compile_stage = "tests"
        link_command = [
            compiler,
            *flags,
            str(combined),
            "-o",
            str(executable),
            "-lm",
        ]
        try:
            proc = subprocess.run(
                link_command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            result.timed_out = True
            result.stderr = str(exc)
            return result
        result.compile_returncode = proc.returncode
        result.stderr = proc.stderr[-4000:]
        if proc.returncode != 0:
            # ``compilable`` follows the historical metric and remains true
            # when the standalone candidate compiled, even if it does not link
            # against the benchmark harness (for example, due to a wrong
            # signature).  Re-executability remains false.
            return result

        result.compile_stage = "complete"
        try:
            proc = subprocess.run(
                [str(executable)],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                cwd=tmpdir,
            )
            result.run_returncode = proc.returncode
            result.stderr = (result.stderr + "\n" + proc.stderr)[-4000:]
            result.re_executable = proc.returncode == 0
        except subprocess.TimeoutExpired as exc:
            result.timed_out = True
            result.stderr = (result.stderr + "\n" + str(exc))[-4000:]
    return result
