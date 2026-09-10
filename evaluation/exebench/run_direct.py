#!/usr/bin/env python3
"""Run and score a direct DecIR/LLM4Decompile checkpoint on ExeBench."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
DATASET = PROJECT / "decompile-eval/exebench-test-real-c-o0-o3.json"
METADATA = PROJECT / "decompile-eval/exebench-test-real-c-metadata.json"


def read_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_shard_rows(
    rows: list[dict], shard_index: int, shards: int
) -> list[dict]:
    return [
        row
        for index, row in enumerate(rows)
        if (index // 4) % shards == shard_index
    ]


def complete(path: Path, expected: list[dict], model: Path) -> bool:
    if not path.is_file():
        return False
    try:
        rows = read_rows(path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        [row.get("task_id") for row in rows]
        == [row["task_id"] for row in expected]
        and all(
            observed.get("input_asm_prompt")
            == reference["input_asm_prompt"]
            for observed, reference in zip(rows, expected)
        )
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
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--objcopy", default="objcopy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    args = parser.parse_args()

    if not args.model.is_dir():
        parser.error(f"model not found: {args.model}")
    if not args.dataset.is_file():
        parser.error(
            f"dataset not found: {args.dataset}; run evaluation/exebench/prepare.py"
        )
    if not args.metadata.is_file():
        parser.error(f"execution metadata not found: {args.metadata}")
    predictions = args.predictions or (
        PROJECT / f"results/predictions/{args.name}-exebench-c.json"
    )
    metrics = args.metrics or (
        PROJECT / f"results/metrics/{args.name}-exebench-c.json"
    )
    rows = read_rows(args.dataset)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one GPU")
    shard_paths = [
        predictions.with_name(
            f"{predictions.stem}.shard-{index}-of-{len(gpus)}.json"
        )
        for index in range(len(gpus))
    ]
    if args.force or not complete(predictions, rows, args.model):
        jobs = []
        for index, (gpu, shard) in enumerate(zip(gpus, shard_paths)):
            shard_expected = expected_shard_rows(rows, index, len(gpus))
            if not args.force and complete(shard, shard_expected, args.model):
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
                        "4",
                    ],
                    gpu,
                )
            )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(gpus)
        ) as executor:
            futures = [
                executor.submit(run, command, gpu) for command, gpu in jobs
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
                *shard_paths,
                "--output",
                predictions,
            ]
        )
    else:
        print(f"Reusing complete predictions: {predictions}")
    if not args.skip_score:
        run(
            [
                sys.executable,
                "evaluation/exebench/evaluate.py",
                "--dataset",
                args.dataset,
                "--metadata",
                args.metadata,
                "--predictions",
                predictions,
                "--output",
                metrics,
                "--workers",
                str(args.workers),
                "--cc",
                args.cc,
                "--cxx",
                args.cxx,
                "--objcopy",
                args.objcopy,
                "--bootstrap-samples",
                "10000",
                "--seed",
                str(args.seed),
            ]
        )


if __name__ == "__main__":
    main()
