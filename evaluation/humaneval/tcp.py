#!/usr/bin/env python3
"""Compute per-Goron-pass TCP for all reproduced HumanEval models."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from _common import METRICS, PREDICTIONS, run


@dataclass(frozen=True)
class Job:
    model: str
    prediction: Path
    output: Path
    prediction_field: str = "output"
    partial_optimizations: bool = False


JOBS = (
    Job(
        "DecIR-1.3B",
        PREDICTIONS / "decir-1.3b-humaneval-goron.json",
        METRICS / "tcp/decir-1.3b-humaneval-goron.json",
    ),
    Job(
        "DecIR-6.7B",
        PREDICTIONS / "decir-6.7b-humaneval-goron.json",
        METRICS / "tcp/decir-6.7b-humaneval-goron.json",
    ),
    Job(
        "LLM4Decompile-1.3B-v1.5",
        (
            PREDICTIONS
            / "llm4decompile-1.3b-v1.5-humaneval-goron.json"
        ),
        (
            METRICS
            / "tcp/llm4decompile-1.3b-v1.5-humaneval-goron.json"
        ),
    ),
    Job(
        "LLM4Decompile-6.7B-v1.5",
        (
            PREDICTIONS
            / "llm4decompile-6.7b-v1.5-humaneval-goron.json"
        ),
        (
            METRICS
            / "tcp/llm4decompile-6.7b-v1.5-humaneval-goron.json"
        ),
    ),
    Job(
        "sc²dec-6.7B",
        (
            PREDICTIONS
            / "sccdec-6.7b-standard-lora-humaneval-goron.json"
        ),
        (
            METRICS
            / "tcp/sccdec-6.7b-standard-lora-humaneval-goron.json"
        ),
    ),
    Job(
        "SLaDe",
        PREDICTIONS / "slade-humaneval-goron-gas-typed.json",
        METRICS / "tcp/slade-humaneval-goron.json",
        prediction_field="typed_candidates",
        partial_optimizations=True,
    ),
    Job(
        "Nova-1.3B",
        PREDICTIONS / "nova-1.3b-humaneval-goron.json",
        METRICS / "tcp/nova-1.3b-humaneval-goron.json",
    ),
    Job(
        "Nova-6.7B",
        PREDICTIONS / "nova-6.7b-humaneval-goron.json",
        METRICS / "tcp/nova-6.7b-humaneval-goron.json",
    ),
)


def main() -> None:
    for job in JOBS:
        command: list[str | Path] = [
            sys.executable,
            "evaluation/evaluate_tcp.py",
            "--benchmark",
            "humaneval",
            "--model-name",
            job.model,
            "--input-compiler",
            "Goron Clang 7.1.0",
            "--predictions",
            job.prediction,
            "--prediction-field",
            job.prediction_field,
            "--output",
            job.output,
            "--compiler",
            "gcc",
            "--workers",
            "32",
        ]
        if job.partial_optimizations:
            command.append("--allow-partial-optimizations")
        run(command)


if __name__ == "__main__":
    main()
