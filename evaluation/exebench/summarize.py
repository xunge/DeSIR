#!/usr/bin/env python3
"""Aggregate all seven ExeBench C/O0--O3 executable metric artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
OPTIMIZATIONS = ("O0", "O1", "O2", "O3")


@dataclass(frozen=True)
class Model:
    name: str
    metrics: Path
    protocol: str
    latex: str


MODELS = (
    Model(
        "DecIR-1.3B",
        PROJECT / "results/metrics/decir-1.3b-exebench-c.json",
        "greedy direct generation",
        "DecIR-1.3B",
    ),
    Model(
        "DecIR-6.7B",
        PROJECT / "results/metrics/decir-6.7b-exebench-c.json",
        "greedy direct generation",
        "DecIR-6.7B",
    ),
    Model(
        "LLM4Decompile-1.3B-v1.5",
        PROJECT
        / "results/metrics/llm4decompile-1.3b-v1.5-exebench-c.json",
        "greedy direct generation",
        "LLM4Decompile-1.3B-v1.5",
    ),
    Model(
        "LLM4Decompile-6.7B-v1.5",
        PROJECT
        / "results/metrics/llm4decompile-6.7b-v1.5-exebench-c.json",
        "greedy direct generation",
        "LLM4Decompile-6.7B-v1.5",
    ),
    Model(
        "sc2dec-6.7B",
        PROJECT
        / (
            "results/metrics/"
            "sccdec-6.7b-standard-lora-exebench-c-final.json"
        ),
        (
            "released FAE LoRA with ordinary alpha/r=2 scaling, one-shot "
            "initial generation, and self-constructed-context second pass"
        ),
        r"sc$^2$dec-6.7B",
    ),
    Model(
        "SLaDe",
        PROJECT / "results/metrics/slade-exebench-c.json",
        (
            "released x86 O0/O3 checkpoints; O1/O2 explicitly use "
            "zero-shot O3-checkpoint transfer"
        ),
        r"SLaDe$^\dagger$",
    ),
    Model(
        "Nova-1.3B",
        PROJECT / "results/metrics/nova-1.3b-exebench-c.json",
        (
            "official address-aware preprocessing and one seeded sampled "
            "candidate (uniform top-1 comparison)"
        ),
        r"Nova-1.3B$^\ddagger$",
    ),
)


def read_model(model: Model, expected_per_level: int) -> dict:
    payload = json.loads(model.metrics.read_text(encoding="utf-8"))
    summary = payload["summary"]
    levels = summary["by_optimization"]
    if any(levels[opt]["total"] != expected_per_level for opt in OPTIMIZATIONS):
        raise ValueError(f"{model.name} does not cover the canonical scope")
    return {
        "model": model.name,
        "metrics": str(model.metrics),
        "protocol": model.protocol,
        "overall": summary["overall"],
        "by_optimization": {opt: levels[opt] for opt in OPTIMIZATIONS},
        "cluster_bootstrap_95_ci": payload["cluster_bootstrap_95_ci"],
    }


def pair(values: dict) -> str:
    return (
        f"{values['compile_rate']:.2f}/"
        f"{values['function_pass_rate']:.2f}"
    )


def write_csv(path: Path, models: list[dict]) -> None:
    fields = ["model"]
    for opt in OPTIMIZATIONS:
        fields.extend(
            [
                f"{opt}_compile_rate",
                f"{opt}_function_pass_rate",
                f"{opt}_io_accuracy",
            ]
        )
    fields.extend(
        [
            "average_compile_rate",
            "average_function_pass_rate",
            "average_io_accuracy",
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model in models:
            row: dict[str, str | float] = {"model": model["model"]}
            for opt in OPTIMIZATIONS:
                values = model["by_optimization"][opt]
                for metric in (
                    "compile_rate",
                    "function_pass_rate",
                    "io_accuracy",
                ):
                    row[f"{opt}_{metric}"] = values[metric]
            for metric in (
                "compile_rate",
                "function_pass_rate",
                "io_accuracy",
            ):
                row[f"average_{metric}"] = model["overall"][metric]
            writer.writerow(row)


def write_markdown(path: Path, models: list[dict]) -> None:
    lines = [
        "# ExeBench test_real C-only executable comparison",
        "",
        (
            "Each O-level cell is compilation rate / function pass rate (%). "
            "A function passes only when all 10 official real I/O tests match."
        ),
        "",
        "| Model | O0 | O1 | O2 | O3 | Avg. Comp/Func | Avg. I/O |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        lines.append(
            "| "
            + " | ".join(
                [
                    model["model"],
                    *(
                        pair(model["by_optimization"][opt])
                        for opt in OPTIMIZATIONS
                    ),
                    pair(model["overall"]),
                    f"{model['overall']['io_accuracy']:.2f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Individual-I/O accuracy (%):",
            "",
            "| Model | O0 | O1 | O2 | O3 | Average |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                "| "
                + " | ".join(
                    [
                        model["model"],
                        *(
                            f"{model['by_optimization'][opt]['io_accuracy']:.2f}"
                            for opt in OPTIMIZATIONS
                        ),
                        f"{model['overall']['io_accuracy']:.2f}",
                    ]
                )
                + " |"
                for model in models
            ],
            "",
            "Protocol notes:",
            "",
            *[
                f"- {model['model']}: {model['protocol']}."
                for model in models
            ],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, models: list[dict]) -> None:
    latex = {model.name: model.latex for model in MODELS}
    lines = [
        "% Auto-generated by project/evaluation/exebench/summarize.py.",
    ]
    for model in models:
        lines.append(
            " & ".join(
                [
                    latex[model["model"]],
                    *(
                        pair(model["by_optimization"][opt])
                        for opt in OPTIMIZATIONS
                    ),
                    pair(model["overall"]),
                    f"{model['overall']['io_accuracy']:.2f}",
                ]
            )
            + r" \\"
        )
    # booktabs rules must be expanded while the input file is still active;
    # placing \bottomrule immediately after \input in the parent tabular makes
    # TeX report a misplaced \noalign.
    lines.append(r"\bottomrule")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT
            / "decompile-eval/exebench-test-real-c-o0-o3.manifest.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT / "results/metrics/exebench-c-model-comparison.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT / "results/metrics/exebench-c-model-comparison.csv",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=PROJECT / "results/metrics/exebench-c-model-comparison.md",
    )
    parser.add_argument(
        "--output-latex",
        type=Path,
        default=PROJECT.parent / "paper/generated_exebench_c_by_opt.tex",
    )
    args = parser.parse_args()
    missing = [str(model.metrics) for model in MODELS if not model.metrics.is_file()]
    if missing:
        parser.error("missing metric artifacts: " + ", ".join(missing))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = manifest["valid_functions"]
    models = [read_model(model, expected) for model in MODELS]
    payload = {
        "benchmark": "ExeBench v1.01 test_real",
        "language": "C",
        "underlying_functions": expected,
        "instances": expected * 4,
        "instances_per_optimization": expected,
        "function_correctness": "all 10 official real I/O pairs pass",
        "models": models,
    }
    for path in (
        args.output_json,
        args.output_csv,
        args.output_markdown,
        args.output_latex,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(args.output_csv, models)
    write_markdown(args.output_markdown, models)
    write_latex(args.output_latex, models)
    print(f"Wrote seven-model comparison to {args.output_json}")


if __name__ == "__main__":
    main()
