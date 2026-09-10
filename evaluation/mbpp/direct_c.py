#!/usr/bin/env python3
"""Run a direct causal-LM checkpoint on MBPP-Decompile C/O0--O3."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
FULL_DATASET = PROJECT / "decompile-eval/mbpp-decompile.json"
C_DATASET = PROJECT / "decompile-eval/mbpp-decompile-c.json"
REFERENCE_VALIDATION = (
    PROJECT / "results/metrics/mbpp-decompile-reference-validation.json"
)
DEFAULT_MODEL = Path("/home/jiang/model/llm4decompile-6.7b-v1.5")
DEFAULT_PREDICTIONS = (
    PROJECT
    / "results/predictions/llm4decompile-6.7b-v1.5-mbpp-c.json"
)
DEFAULT_METRICS = (
    PROJECT / "results/metrics/llm4decompile-6.7b-v1.5-mbpp-c.json"
)
OPTIMIZATIONS = ("O0", "O1", "O2", "O3")


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def prepare_c_dataset(source: Path, output: Path) -> list[dict]:
    source_rows = load_rows(source)
    rows = [row for row in source_rows if row.get("language") == "c"]
    task_ids = [row["task_id"] for row in rows]
    optimization_counts = Counter(row["type"] for row in rows)
    if (
        len(rows) != 3896
        or len(task_ids) != len(set(task_ids))
        or optimization_counts != Counter({opt: 974 for opt in OPTIMIZATIONS})
        or len({row["cluster_id"] for row in rows}) != 974
    ):
        raise ValueError(
            "expected 3,896 unique C rows, 974 clusters, and 974 rows "
            f"per optimization level; observed rows={len(rows)}, "
            f"optimizations={dict(optimization_counts)}"
        )
    serialized = json.dumps(rows, indent=2, ensure_ascii=False)
    if not output.is_file() or output.read_text(encoding="utf-8") != serialized:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        manifest = {
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "language": "c",
            "instances": len(rows),
            "underlying_clusters": 974,
            "optimization_levels": dict(sorted(optimization_counts.items())),
        }
        output.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    return rows


def expected_shard_ids(
    rows: list[dict], shard_index: int, shards: int, group_size: int
) -> list[str]:
    return [
        row["task_id"]
        for index, row in enumerate(rows)
        if (index // group_size) % shards == shard_index
    ]


def prediction_complete(
    path: Path, expected_ids: list[str], model: Path
) -> bool:
    if not path.is_file():
        return False
    try:
        rows = load_rows(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    observed_ids = [row.get("task_id") for row in rows]
    expected_model = model.resolve()
    return (
        observed_ids == expected_ids
        and all("output" in row for row in rows)
        and all(
            Path(row.get("generation", {}).get("model", "")).resolve()
            == expected_model
            for row in rows
        )
    )


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
    subprocess.run(rendered, cwd=PROJECT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source-dataset", type=Path, default=FULL_DATASET)
    parser.add_argument("--dataset", type=Path, default=C_DATASET)
    parser.add_argument(
        "--reference-validation",
        type=Path,
        default=REFERENCE_VALIDATION,
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    args = parser.parse_args()

    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    rows = prepare_c_dataset(args.source_dataset, args.dataset)
    expected_ids = [row["task_id"] for row in rows]
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one GPU ID")
    group_size = len(OPTIMIZATIONS)
    shard_paths = [
        args.predictions.with_name(
            f"{args.predictions.stem}.shard-{index}-of-{len(gpus)}.json"
        )
        for index in range(len(gpus))
    ]

    if args.force_inference or not prediction_complete(
        args.predictions, expected_ids, args.model
    ):
        jobs: list[tuple[list[str | Path], str]] = []
        for shard_index, (gpu, shard_path) in enumerate(
            zip(gpus, shard_paths)
        ):
            shard_ids = expected_shard_ids(
                rows, shard_index, len(gpus), group_size
            )
            if not args.force_inference and prediction_complete(
                shard_path, shard_ids, args.model
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
                        str(group_size),
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
            args.source_dataset,
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


if __name__ == "__main__":
    main()
