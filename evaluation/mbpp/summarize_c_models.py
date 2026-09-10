#!/usr/bin/env python3
"""Summarize MBPP-Decompile C/O0--O3 metrics across local checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
OPTIMIZATIONS = ("O0", "O1", "O2", "O3")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    metrics: Path
    protocol: str
    latex_name: str


MODELS = (
    ModelSpec(
        "DecIR-1.3B",
        PROJECT / "results/metrics/decir-1.3b-mbpp-decompile.json",
        "direct raw assembly; greedy first candidate",
        "DecIR-1.3B",
    ),
    ModelSpec(
        "DecIR-6.7B",
        PROJECT / "results/metrics/decir-6.7b-mbpp-decompile.json",
        "direct raw assembly; greedy first candidate",
        "DecIR-6.7B",
    ),
    ModelSpec(
        "LLM4Decompile-1.3B-v1.5",
        PROJECT
        / "results/metrics/llm4decompile-1.3b-v1.5-mbpp-decompile.json",
        "direct raw assembly; greedy first candidate",
        "LLM4Decompile-1.3B-v1.5",
    ),
    ModelSpec(
        "LLM4Decompile-6.7B-v1.5",
        PROJECT
        / "results/metrics/llm4decompile-6.7b-v1.5-mbpp-c.json",
        "direct raw assembly; greedy first candidate",
        "LLM4Decompile-6.7B-v1.5",
    ),
    ModelSpec(
        "sc2dec-6.7B",
        PROJECT
        / (
            "results/metrics/"
            "sccdec-6.7b-standard-lora-mbpp-c-final.json"
        ),
        (
            "FAE LoRA with ordinary alpha/r=2 scaling, one-shot initial "
            "generation, and dependency-aware self-constructed-context "
            "second pass"
        ),
        "sc$^2$dec-6.7B$^{\\dagger}$",
    ),
    ModelSpec(
        "SLaDe",
        PROJECT
        / "results/metrics/slade-released-x86-allopts-mbpp-c.json",
        (
            "released x86 O0/O3 checkpoints and GCC/GAS input; "
            "O1/O2 are zero-shot transfer through the O3 checkpoint"
        ),
        "SLaDe$^{\\ddagger}$",
    ),
    ModelSpec(
        "Nova-1.3B",
        PROJECT / "results/metrics/nova-1.3b-mbpp-c-first-sample.json",
        "official address-aware normalization; first of 20 sampled candidates",
        "Nova-1.3B$^{\\S}$",
    ),
)


def read_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "summary" not in payload or "per_case" not in payload:
        raise ValueError(f"{path} is not an executable metric artifact")
    return payload


def c_summary(payload: dict) -> tuple[dict, dict]:
    summary = payload["summary"]
    languages = {row["language"] for row in payload["per_case"]}
    if languages == {"c"}:
        return summary["overall"], summary["by_optimization"]
    if "c" not in summary.get("by_language", {}):
        raise ValueError("metric artifact has no C results")
    return (
        summary["by_language"]["c"],
        summary["by_language_and_optimization"]["c"],
    )


def validate_model(spec: ModelSpec, payload: dict) -> dict:
    overall, levels = c_summary(payload)
    missing = [opt for opt in OPTIMIZATIONS if opt not in levels]
    if missing:
        raise ValueError(f"{spec.name} lacks optimization levels: {missing}")
    unexpected_totals = {
        opt: levels[opt]["total"]
        for opt in OPTIMIZATIONS
        if levels[opt]["total"] != 974
    }
    if unexpected_totals or overall["total"] != 3896:
        raise ValueError(
            f"{spec.name} does not cover the canonical 3,896-row C scope: "
            f"overall={overall['total']}, levels={unexpected_totals}"
        )
    return {
        "model": spec.name,
        "metrics": str(spec.metrics),
        "protocol": spec.protocol,
        "overall": {
            key: overall[key]
            for key in (
                "total",
                "compile_count",
                "run_count",
                "compile_rate",
                "run_rate",
                "edit_similarity",
            )
        },
        "by_optimization": {
            opt: {
                key: levels[opt][key]
                for key in (
                    "total",
                    "compile_count",
                    "run_count",
                    "compile_rate",
                    "run_rate",
                    "edit_similarity",
                )
            }
            for opt in OPTIMIZATIONS
        },
    }


def write_csv(path: Path, models: list[dict]) -> None:
    fields = ["model"]
    for opt in OPTIMIZATIONS:
        fields.extend(
            [
                f"{opt}_compile_rate",
                f"{opt}_run_rate",
                f"{opt}_edit_similarity",
            ]
        )
    fields.extend(
        ["average_compile_rate", "average_run_rate", "average_edit_similarity"]
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model in models:
            row: dict[str, str | float] = {"model": model["model"]}
            for opt in OPTIMIZATIONS:
                values = model["by_optimization"][opt]
                row[f"{opt}_compile_rate"] = values["compile_rate"]
                row[f"{opt}_run_rate"] = values["run_rate"]
                row[f"{opt}_edit_similarity"] = values["edit_similarity"]
            row["average_compile_rate"] = model["overall"]["compile_rate"]
            row["average_run_rate"] = model["overall"]["run_rate"]
            row["average_edit_similarity"] = model["overall"]["edit_similarity"]
            writer.writerow(row)


def pair(values: dict) -> str:
    return f"{values['compile_rate']:.2f}/{values['run_rate']:.2f}"


def write_markdown(path: Path, models: list[dict]) -> None:
    lines = [
        "# MBPP-Decompile C-only comparison",
        "",
        "Each cell is RC/RE (%); Avg Edit is normalized edit similarity (%).",
        "",
        "| Model | O0 | O1 | O2 | O3 | Avg. RC/RE | Avg. Edit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        levels = model["by_optimization"]
        lines.append(
            "| "
            + " | ".join(
                [
                    model["model"],
                    *(pair(levels[opt]) for opt in OPTIMIZATIONS),
                    pair(model["overall"]),
                    f"{model['overall']['edit_similarity']:.2f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
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
    latex_names = {spec.name: spec.latex_name for spec in MODELS}
    lines = [
        "% Auto-generated by project/evaluation/mbpp/summarize_c_models.py.",
    ]
    for model in models:
        levels = model["by_optimization"]
        fields = [
            latex_names[model["model"]],
            *(pair(levels[opt]) for opt in OPTIMIZATIONS),
            pair(model["overall"]),
            f"{model['overall']['edit_similarity']:.2f}",
        ]
        lines.append(" & ".join(fields) + r" \\")
    lines.append(r"\bottomrule")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT / "results/metrics/mbpp-c-model-comparison.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT / "results/metrics/mbpp-c-model-comparison.csv",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=PROJECT / "results/metrics/mbpp-c-model-comparison.md",
    )
    parser.add_argument(
        "--output-latex",
        type=Path,
        default=PROJECT.parent / "paper/generated_mbpp_c_by_opt.tex",
    )
    args = parser.parse_args()

    missing = [str(spec.metrics) for spec in MODELS if not spec.metrics.is_file()]
    if missing:
        parser.error("missing metric artifacts: " + ", ".join(missing))
    models = [validate_model(spec, read_metrics(spec.metrics)) for spec in MODELS]
    payload = {
        "benchmark": "MBPP-Decompile",
        "language": "c",
        "instances": 3896,
        "underlying_problems": 974,
        "instances_per_optimization": 974,
        "cell_format": "percent",
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
    print(f"Wrote {len(models)}-model C-only summary to {args.output_json}")


if __name__ == "__main__":
    main()
