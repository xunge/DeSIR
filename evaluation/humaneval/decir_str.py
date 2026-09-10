#!/usr/bin/env python3
"""Evaluate a string-aware DecIR checkpoint on Clang HumanEval-Decompile.

The input and decoding protocol follows ``evaluation_string/decompile_binary.py``:
compile each reference function with Clang, recover binary-grounded strings,
annotate the disassembly with per-function ``STR_i`` tokens, greedily generate
C, and deterministically restore the strings.  Generated candidates are then
scored with the repository's common GNU C RC/RE and function-macro TCP oracles.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from types import SimpleNamespace


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM = Path(
    "/home/jiang/projects/DecIR/evaluation_string/"
    "run_evaluation_LIRAD_vllm.py"
)
OPTIMIZATIONS = ("O0", "O1", "O2", "O3")

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from evaluation.evaluate_decompile_bench import normalized_edit_similarity


def model_display_name(args) -> str:
    if args.model_name:
        return args.model_name
    name = re.sub(r"(?i)^decir", "DecIR", args.model.name)
    return re.sub(r"(?i)(\d+(?:\.\d+)?)b", r"\1B", name)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_upstream(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"string-aware evaluator not found: {path}")
    spec = importlib.util.spec_from_file_location("decir_string_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import string-aware evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_source_dataset(path: Path) -> list[dict]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"dataset must be a JSON list: {path}")
    counts = Counter(str(row.get("type")) for row in rows)
    expected = Counter({optimization: 164 for optimization in OPTIMIZATIONS})
    if len(rows) != 656 or counts != expected:
        raise ValueError(
            f"expected 656 HumanEval rows and 164 per optimization; "
            f"observed {len(rows)} rows and {dict(counts)}"
        )
    return rows


def predictions_complete(path: Path, model: Path) -> bool:
    if not path.is_file():
        return False
    try:
        rows = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    counts = Counter(str(row.get("type")) for row in rows)
    return (
        len(rows) == 656
        and counts
        == Counter({optimization: 164 for optimization in OPTIMIZATIONS})
        and all("output" in row and "output_restored" in row for row in rows)
        and all(
            Path(str(row.get("generation", {}).get("model", ""))).resolve()
            == model.resolve()
            for row in rows
        )
    )


def command(parts: list[str | Path], *, env: dict[str, str] | None = None) -> None:
    rendered = [str(part) for part in parts]
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print("+", " ".join(rendered), flush=True)
    subprocess.run(rendered, cwd=PROJECT, env=merged, check=True)


def prepare_and_generate(args) -> None:
    # CUDA visibility must be fixed before the upstream helper lazily imports
    # vLLM/torch.  A single 96 GB GPU is sufficient for this 6.7B checkpoint.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    upstream = load_upstream(args.upstream_evaluator)
    upstream_args = SimpleNamespace(
        model_path=str(args.model),
        tokenizer_path=(str(args.tokenizer) if args.tokenizer else None),
        compiler="clang",
        gpus=1,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=0.0,
        max_total_tokens=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        repeat=1,
        testset_path=str(args.dataset),
        output_path=str(args.predictions),
        output_result_path=None,
        num_workers=args.workers,
        reuse_outputs=False,
        string_prompt_mode="auto",
        prepared_testset_output=str(args.prepared_dataset),
        string_threshold=args.string_threshold,
        min_string_length=args.min_string_length,
        max_scan_bytes=args.max_scan_bytes,
        prompt_workers=args.prompt_workers,
        compile_timeout=args.compile_timeout,
        run_timeout=args.run_timeout,
    )
    context = upstream.prepare_eval_context(upstream_args)
    llm = None
    try:
        llm = upstream.build_llm(upstream_args)
        generated = upstream.generate_results(
            llm,
            context["inputs"],
            context["stop_sequences"],
            upstream_args,
        )[0]
    finally:
        if llm is not None:
            upstream.cleanup_vllm(llm)

    output = []
    for row, result in zip(context["testsets"], generated):
        raw = str(result[0])
        output.append(
            {
                **row,
                "output": raw,
                "output_restored": upstream.restore_generated_code(
                    raw, row, "clang"
                ),
                "generation": {
                    "model": str(args.model.resolve()),
                    "tokenizer": str(
                        (args.tokenizer or args.model).resolve()
                    ),
                    "input_compiler": "clang",
                    "prompt_protocol": "binary-grounded STR_i annotations",
                    "temperature": 0.0,
                    "max_model_len": args.max_model_len,
                    "max_new_tokens": args.max_new_tokens,
                    "string_threshold": args.string_threshold,
                },
            }
        )
    if len(output) != 656:
        raise RuntimeError(f"generation coverage mismatch: {len(output)} != 656")
    write_json(args.predictions, output)
    print(f"Wrote {len(output)} restored predictions to {args.predictions}")


def summarize(args) -> dict:
    from transformers import AutoTokenizer

    predictions = load_json(args.predictions)
    metrics = load_json(args.metrics)
    tcp = load_json(args.tcp_metrics)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    prompt_prefix = "# This is the assembly code:\n"
    prompt_suffix = "\n# What is the source code?\n"
    prompt_lengths = [
        len(
            tokenizer.encode(
                prompt_prefix
                + str(row["input_asm_prompt"]).strip()
                + prompt_suffix,
                add_special_tokens=True,
            )
        )
        for row in predictions
    ]
    output_lengths = [
        len(tokenizer.encode(str(row["output"]), add_special_tokens=False))
        for row in predictions
    ]
    grouped: dict[str, list[float]] = defaultdict(list)
    string_rows = 0
    string_references = 0
    string_outputs = 0
    restored_outputs = 0
    unresolved_outputs = 0
    for row in predictions:
        optimization = str(row["type"])
        candidate = str(row["output_restored"])
        grouped[optimization].append(
            100
            * normalized_edit_similarity(str(row.get("c_func", "")), candidate)
        )
        recovery = row.get("string_refs", [])
        if recovery:
            string_rows += 1
            string_references += len(recovery)
        if "STR_" in str(row["output"]):
            string_outputs += 1
        if row["output"] != row["output_restored"]:
            restored_outputs += 1
        if "STR_" in candidate:
            unresolved_outputs += 1

    rows = []
    for optimization in OPTIMIZATIONS:
        legacy = metrics["summary"][optimization]
        tcp_level = tcp["summary"]["by_optimization"][optimization]
        rows.append(
            {
                "optimization": optimization,
                "functions": legacy["total"],
                "rc": legacy["compile_rate"],
                "re": legacy["run_rate"],
                "edit": round(
                    sum(grouped[optimization]) / len(grouped[optimization]), 2
                ),
                "tcp": tcp_level["tcp"],
                "tcp_valid_functions": tcp_level["functions"],
            }
        )
    result = {
        "model": model_display_name(args),
        "model_path": str(args.model.resolve()),
        "benchmark": "HumanEval-Decompile",
        "input_compiler": "Clang 18.1.3",
        "candidate_oracle": "GCC 13.3.0 GNU C",
        "protocol": {
            "input": (
                "binary-derived function assembly with referenced strings "
                "annotated as STR_i"
            ),
            "decoding": "greedy first candidate, temperature=0",
            "string_restoration": "deterministic per-function STR_i expansion",
            "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
        },
        "coverage": {
            "predictions": len(predictions),
            "functions_per_optimization": 164,
            "tcp_valid_functions": tcp["valid_functions"],
            "tcp_excluded_invalid_oracle_functions": tcp[
                "excluded_invalid_oracle_functions"
            ],
            "rows_with_referenced_strings": string_rows,
            "total_string_references": string_references,
            "outputs_using_STR_tokens": string_outputs,
            "outputs_changed_by_string_restoration": restored_outputs,
            "outputs_with_unresolved_STR_tokens": unresolved_outputs,
            "maximum_prompt_tokens": max(prompt_lengths),
            "context_overflows": sum(
                length > args.max_model_len - args.max_new_tokens
                for length in prompt_lengths
            ),
            "outputs_at_1024_token_cap": sum(
                length >= args.max_new_tokens for length in output_lengths
            ),
        },
        "by_optimization": rows,
        "average": {
            "rc": metrics["summary"]["mean"]["compile_rate"],
            "re": metrics["summary"]["mean"]["run_rate"],
            "edit": round(
                sum(value for values in grouped.values() for value in values)
                / len(predictions),
                2,
            ),
            "tcp": tcp["summary"]["macro_average_of_optimization_tcp"],
        },
        "artifacts": {
            "prepared_dataset": str(args.prepared_dataset),
            "predictions": str(args.predictions),
            "rc_re_metrics": str(args.metrics),
            "tcp_metrics": str(args.tcp_metrics),
        },
    }
    write_json(args.summary, result)
    table = [
        f"# {model_display_name(args)} on HumanEval-Decompile (Clang input)",
        "",
        "| Optimization | Functions | RC | RE | Edit | TCP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['optimization']} | {row['functions']} | "
            f"{row['rc']:.2f} | {row['re']:.2f} | {row['edit']:.2f} | "
            f"{row['tcp']:.2f} |"
        )
    average = result["average"]
    table.append(
        f"| Average | 656 | {average['rc']:.2f} | {average['re']:.2f} | "
        f"{average['edit']:.2f} | {average['tcp']:.2f} |"
    )
    table.extend(
        [
            "",
            "Clang 18.1.3 generates the string-aware model inputs; GCC 13.3.0 "
            "is the candidate compilation/execution oracle, matching the attached "
            "evaluation implementation.",
        ]
    )
    args.summary.with_suffix(".md").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("/home/jiang/model/decir-6.7b-str")
    )
    parser.add_argument(
        "--model-name",
        help="display name stored in TCP and summary artifacts",
    )
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT
        / "decompile-eval/decompile-eval-executable-clang-obj.json",
    )
    parser.add_argument(
        "--prepared-dataset",
        type=Path,
        default=PROJECT
        / "decompile-eval/humaneval-decompile-clang-string-aware.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT
        / "results/predictions/decir-6.7b-str-humaneval-clang.json",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT
        / "results/metrics/decir-6.7b-str-humaneval-clang.json",
    )
    parser.add_argument(
        "--tcp-metrics",
        type=Path,
        default=PROJECT
        / "results/metrics/tcp/decir-6.7b-str-humaneval-clang.json",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT
        / "results/metrics/decir-6.7b-str-humaneval-clang-summary.json",
    )
    parser.add_argument("--upstream-evaluator", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--prompt-workers", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--string-threshold", type=int, default=128)
    parser.add_argument("--min-string-length", type=int, default=2)
    parser.add_argument("--max-scan-bytes", type=int, default=1048576)
    parser.add_argument("--compile-timeout", type=int, default=10)
    parser.add_argument("--run-timeout", type=int, default=10)
    parser.add_argument(
        "--tcp-repeats",
        type=int,
        default=3,
        help=(
            "Repeat TCP execution and retain the run with the median aggregate "
            "TCP. This limits noise from undefined behavior in generated C."
        ),
    )
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(f"model directory not found: {args.model}")
    validate_source_dataset(args.dataset)
    if args.force_inference and args.score_only:
        raise ValueError("--force-inference and --score-only are mutually exclusive")
    complete = predictions_complete(args.predictions, args.model)
    if args.score_only and not complete:
        raise RuntimeError(f"complete predictions not found: {args.predictions}")
    if args.force_inference or not complete:
        prepare_and_generate(args)
    else:
        print(f"Reusing complete predictions: {args.predictions}")

    python = sys.executable
    command(
        [
            python,
            "evaluation/evaluate_predictions.py",
            "--predictions",
            args.predictions,
            "--prediction-field",
            "output_restored",
            "--output",
            args.metrics,
            "--compiler",
            "gcc",
            "--workers",
            str(args.workers),
            "--no-strict",
            "--bootstrap-samples",
            "10000",
            "--seed",
            "42",
        ]
    )
    if args.tcp_repeats < 1 or args.tcp_repeats % 2 == 0:
        raise ValueError("--tcp-repeats must be a positive odd number")
    tcp_runs = []
    for repeat_index in range(args.tcp_repeats):
        repeat_output = (
            args.tcp_metrics
            if args.tcp_repeats == 1
            else args.tcp_metrics.with_name(
                f"{args.tcp_metrics.stem}-repeat{repeat_index + 1}"
                f"{args.tcp_metrics.suffix}"
            )
        )
        command(
            [
                python,
                "evaluation/evaluate_tcp.py",
                "--benchmark",
                "humaneval",
                "--model-name",
                model_display_name(args),
                "--input-compiler",
                "Clang 18.1.3 string-aware binary",
                "--predictions",
                args.predictions,
                "--prediction-field",
                "output_restored",
                "--output",
                repeat_output,
                "--compiler",
                "gcc",
                "--workers",
                str(args.workers),
            ]
        )
        payload = load_json(repeat_output)
        tcp_runs.append(
            (
                float(
                    payload["summary"][
                        "macro_average_of_optimization_tcp"
                    ]
                ),
                repeat_output,
                payload,
            )
        )
    if args.tcp_repeats > 1:
        median_tcp = median(value for value, _, _ in tcp_runs)
        selected = min(
            tcp_runs,
            key=lambda item: (abs(item[0] - median_tcp), str(item[1])),
        )
        selected[2]["protocol"]["repeat_selection"] = {
            "policy": "median aggregate TCP",
            "repeats": args.tcp_repeats,
            "aggregate_tcp_values": [value for value, _, _ in tcp_runs],
            "selected": str(selected[1]),
        }
        write_json(args.tcp_metrics, selected[2])
        print(
            f"Selected median TCP run {selected[1]} "
            f"({selected[0]:.2f}) as {args.tcp_metrics}"
        )
    result = summarize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
