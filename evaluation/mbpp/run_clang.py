#!/usr/bin/env python3
"""Run and score one DecIR checkpoint on linked Clang MBPP C inputs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
OPTIMIZATIONS = ("O0", "O1", "O2", "O3")


def read_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def shard_ids(
    rows: list[dict], shard_index: int, shards: int
) -> list[str]:
    return [
        row["task_id"]
        for index, row in enumerate(rows)
        if (index // len(OPTIMIZATIONS)) % shards == shard_index
    ]


def complete(path: Path, expected: list[str], model: Path) -> bool:
    if not path.is_file():
        return False
    try:
        rows = read_rows(path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        [row.get("task_id") for row in rows] == expected
        and all("output" in row for row in rows)
        and all(
            Path(row.get("generation", {}).get("model", "")).resolve()
            == model.resolve()
            for row in rows
        )
    )


def run(command: list[str | Path], gpu: str | None = None) -> None:
    rendered = [str(part) for part in command]
    print("+", " ".join(rendered), flush=True)
    environment = os.environ.copy()
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    subprocess.run(rendered, cwd=PROJECT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            PROJECT
            / "decompile-eval/mbpp-decompile-clang-linked-c.json"
        ),
    )
    parser.add_argument(
        "--reference-validation",
        type=Path,
        default=(
            PROJECT
            / "results/metrics/mbpp-decompile-reference-validation.json"
        ),
    )
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--score-cc", default="gcc")
    parser.add_argument("--score-cxx", default="g++")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.dataset)
    predictions = args.predictions or (
        PROJECT
        / f"results/predictions/{args.name}-mbpp-clang-linked-c.json"
    )
    metrics = args.metrics or (
        PROJECT / f"results/metrics/{args.name}-mbpp-clang-linked-c.json"
    )
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    expected = [row["task_id"] for row in rows]
    shards = [
        predictions.with_name(
            f"{predictions.stem}.shard-{index}-of-{len(gpus)}.json"
        )
        for index in range(len(gpus))
    ]
    if args.force or not complete(predictions, expected, args.model):
        jobs: list[tuple[list[str | Path], str]] = []
        for index, (gpu, shard) in enumerate(zip(gpus, shards)):
            expected_ids = shard_ids(rows, index, len(gpus))
            if not args.force and complete(shard, expected_ids, args.model):
                print(f"Reusing complete shard: {shard}")
                continue
            jobs.append(
                (
                    [
                        sys.executable,
                        "evaluation/run_inference_vllm.py",
                        "--model",
                        args.model,
                        "--dataset",
                        args.dataset,
                        "--output",
                        shard,
                        "--mode",
                        "direct",
                        "--tensor-parallel-size",
                        "1",
                        "--gpu-memory-utilization",
                        str(args.gpu_memory_utilization),
                        "--max-model-len",
                        str(args.max_model_len),
                        "--max-new-tokens",
                        str(args.max_new_tokens),
                        "--temperature",
                        "0",
                        "--top-p",
                        "1",
                        "--seed",
                        str(args.seed),
                        "--shard-index",
                        str(index),
                        "--num-shards",
                        str(len(gpus)),
                        "--shard-group-size",
                        str(len(OPTIMIZATIONS)),
                    ],
                    gpu,
                )
            )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(gpus)
        ) as executor:
            futures = [
                executor.submit(run, command, gpu)
                for command, gpu in jobs
            ]
            for future in futures:
                future.result()
        run(
            [
                sys.executable,
                "evaluation/merge_prediction_shards.py",
                "--dataset",
                args.dataset,
                "--inputs",
                *shards,
                "--output",
                predictions,
            ]
        )
    if args.skip_score:
        return
    run(
        [
            sys.executable,
            "evaluation/evaluate_decompile_bench.py",
            "--predictions",
            predictions,
            "--output",
            metrics,
            "--benchmark-metadata",
            args.dataset,
            "--reference-validation",
            args.reference_validation,
            "--c-compiler",
            args.score_cc,
            "--cpp-compiler",
            args.score_cxx,
            "--workers",
            str(args.workers),
            "--bootstrap-samples",
            "10000",
            "--seed",
            str(args.seed),
            "--crypto-library=-l:libcrypto.so.3",
        ]
    )


if __name__ == "__main__":
    main()
