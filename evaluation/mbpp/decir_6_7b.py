#!/usr/bin/env python3
"""Run and score DecIR-6.7B on the full MBPP-Decompile benchmark."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
DATASET = PROJECT / "decompile-eval/mbpp-decompile.json"
REFERENCE_VALIDATION = (
    PROJECT / "results/metrics/mbpp-decompile-reference-validation.json"
)
PREDICTIONS = (
    PROJECT / "results/predictions/decir-6.7b-mbpp-decompile.json"
)
METRICS = PROJECT / "results/metrics/decir-6.7b-mbpp-decompile.json"


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def prediction_complete(path: Path, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        rows = load_rows(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return len(rows) == expected_rows and all("output" in row for row in rows)


def run(command: list[str | Path], *, gpu: str | None = None) -> None:
    rendered = [str(part) for part in command]
    print("+", " ".join(rendered), flush=True)
    environment = os.environ.copy()
    if gpu is not None:
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            }
        )
    subprocess.run(
        rendered,
        cwd=PROJECT,
        env=environment,
        check=True,
    )


def shard_size(total: int, count: int, index: int, group_size: int) -> int:
    return sum(
        (row // group_size) % count == index for row in range(total)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/jiang/model/decir-6.7b"),
    )
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--reference-validation",
        type=Path,
        default=REFERENCE_VALIDATION,
    )
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="Comma-separated GPU IDs; one inference shard is used per GPU.",
    )
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--shard-group-size",
        type=int,
        default=4,
        help=(
            "Adjacent rows kept together during interleaved sharding; "
            "four preserves every MBPP problem's O0--O3 variants."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    args = parser.parse_args()

    dataset = load_rows(args.dataset)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one GPU ID")
    shard_paths = [
        args.predictions.with_name(
            f"{args.predictions.stem}.shard-{index}-of-{len(gpus)}.json"
        )
        for index in range(len(gpus))
    ]

    if args.force_inference or not prediction_complete(
        args.predictions, len(dataset)
    ):
        jobs = []
        for shard_index, (gpu, shard_path) in enumerate(
            zip(gpus, shard_paths)
        ):
            expected_rows = shard_size(
                len(dataset),
                len(gpus),
                shard_index,
                args.shard_group_size,
            )
            if not args.force_inference and prediction_complete(
                shard_path, expected_rows
            ):
                print(f"Reusing complete shard: {shard_path}", flush=True)
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
                        shard_path,
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
                        str(shard_index),
                        "--num-shards",
                        str(len(gpus)),
                        "--shard-group-size",
                        str(args.shard_group_size),
                    ],
                    gpu,
                )
            )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(gpus)
        ) as executor:
            futures = [
                executor.submit(run, command, gpu=gpu)
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
                *shard_paths,
                "--output",
                args.predictions,
            ]
        )
    else:
        print(f"Reusing complete predictions: {args.predictions}", flush=True)

    if args.skip_score:
        return
    run(
        [
            sys.executable,
            "evaluation/evaluate_decompile_bench.py",
            "--predictions",
            args.predictions,
            "--output",
            args.metrics,
            "--benchmark-metadata",
            args.dataset,
            "--reference-validation",
            args.reference_validation,
            "--workers",
            str(args.workers),
            "--bootstrap-samples",
            "10000",
            "--seed",
            str(args.seed),
            "--crypto-library=-l:libcrypto.so.3",
        ]
    )

    comparisons = [
        (
            PROJECT / "results/metrics/decir-1.3b-mbpp-decompile.json",
            "DecIR-1.3B",
            PROJECT
            / "results/metrics/decir-6.7b-vs-1.3b-mbpp-paired.json",
        ),
        (
            PROJECT
            / (
                "results/metrics/"
                "llm4decompile-1.3b-v1.5-mbpp-decompile.json"
            ),
            "LLM4Decompile-1.3B-v1.5",
            PROJECT
            / (
                "results/metrics/"
                "decir-6.7b-vs-llm4decompile-v1.5-mbpp-paired.json"
            ),
        ),
        (
            PROJECT
            / (
                "results/metrics/"
                "llm4decompile-1.3b-v1.6-mbpp-decompile.json"
            ),
            "LLM4Decompile-1.3B-v1.6",
            PROJECT
            / (
                "results/metrics/"
                "decir-6.7b-vs-llm4decompile-v1.6-mbpp-paired.json"
            ),
        ),
    ]
    for baseline, baseline_name, output in comparisons:
        if not baseline.is_file():
            print(f"Skipping unavailable paired baseline: {baseline}")
            continue
        run(
            [
                sys.executable,
                "evaluation/compare_decompile_bench.py",
                "--left",
                args.metrics,
                "--right",
                baseline,
                "--left-name",
                "DecIR-6.7B",
                "--right-name",
                baseline_name,
                "--output",
                output,
                "--bootstrap-samples",
                "10000",
                "--seed",
                str(args.seed),
            ]
        )
    if all(path.is_file() for _, _, path in comparisons):
        run([sys.executable, "evaluation/mbpp/export_latex.py"])


if __name__ == "__main__":
    main()
