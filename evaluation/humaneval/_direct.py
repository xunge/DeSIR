"""Implementation shared by the DecIR and LLM4Decompile entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (
    DATASET,
    METRICS,
    PREDICTIONS,
    prediction_complete,
    run,
    score_direct,
    vllm_environment,
)


def run_direct(
    *,
    model_name: str,
    default_model: Path,
    prediction_name: str,
    metric_name: str,
) -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Run and score {model_name} on Goron HumanEval-Decompile."
        )
    )
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--predictions", type=Path, default=PREDICTIONS / prediction_name
    )
    parser.add_argument(
        "--metrics", type=Path, default=METRICS / metric_name
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    args = parser.parse_args()

    if args.force_inference or not prediction_complete(
        args.predictions, expected_rows=3936
    ):
        run(
            [
                sys.executable,
                "evaluation/run_inference_vllm.py",
                "--model",
                args.model,
                "--dataset",
                args.dataset,
                "--output",
                args.predictions,
                "--mode",
                "direct",
                "--tensor-parallel-size",
                "1",
                "--gpu-memory-utilization",
                "0.9",
                "--max-model-len",
                "16384",
                "--max-new-tokens",
                "1024",
                "--temperature",
                "0",
                "--top-p",
                "1",
                "--seed",
                "42",
            ],
            env=vllm_environment(args.gpu),
        )
    else:
        print(f"Reusing complete predictions: {args.predictions}")
    if not args.skip_score:
        score_direct(
            args.predictions, args.metrics, workers=args.workers
        )
