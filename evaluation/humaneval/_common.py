"""Shared paths and subprocess helpers for HumanEval evaluation runners."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
PAPER = PROJECT.parent / "paper"
DATASET = PROJECT / "decompile-eval/humaneval-decompile-goron.json"
SOURCE_DATASET = (
    PROJECT / "decompile-eval/decompile-eval-executable-clang-obj.json"
)
SLADE_DATASET = (
    PROJECT / "decompile-eval/humaneval-decompile-goron-slade-gas.json"
)
PREDICTIONS = PROJECT / "results/predictions"
METRICS = PROJECT / "results/metrics"
GORON_CLANG = Path(
    "/home/jiang/projects/open_source/goron/build/bin/clang"
)
NOVA_ROOT = PROJECT.parent / "third_party/nova"
SLADE_ROOT = PROJECT.parent / "third_party/slade_artifact"


def run(
    command: list[str | Path],
    *,
    env: dict[str, str] | None = None,
) -> None:
    rendered = [str(part) for part in command]
    print("+", " ".join(rendered), flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(
        rendered,
        cwd=PROJECT,
        env=merged_env,
        check=True,
    )


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def prediction_complete(
    path: Path,
    *,
    expected_rows: int,
    candidate_field: str = "output",
    candidate_count: int | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        rows = load_rows(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if len(rows) != expected_rows:
        return False
    for row in rows:
        if candidate_field not in row:
            return False
        if candidate_count is not None and (
            not isinstance(row[candidate_field], list)
            or len(row[candidate_field]) != candidate_count
        ):
            return False
    return True


def score_direct(
    predictions: Path,
    output: Path,
    *,
    prediction_field: str = "output",
    workers: int = 32,
) -> None:
    run(
        [
            sys.executable,
            "evaluation/evaluate_predictions.py",
            "--predictions",
            predictions,
            "--prediction-field",
            prediction_field,
            "--output",
            output,
            "--compiler",
            "gcc",
            "--workers",
            str(workers),
            "--no-strict",
            "--bootstrap-samples",
            "10000",
            "--seed",
            "42",
        ]
    )


def vllm_environment(gpu: str) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": gpu,
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    }
