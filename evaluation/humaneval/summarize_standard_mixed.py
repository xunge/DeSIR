#!/usr/bin/env python3
"""Summarize Clang DecIR and GCC baselines on HumanEval-Decompile."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from evaluation.evaluate_decompile_bench import normalized_edit_similarity


PAPER = PROJECT.parent / "paper"
OPTIMIZATIONS = ("O0", "O1", "O2", "O3")
CLANG_DATASET = (
    PROJECT / "decompile-eval/decompile-eval-executable-clang-obj.json"
)
GCC_DATASET = (
    PROJECT / "decompile-eval/decompile-eval-executable-gcc-obj.json"
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    latex_name: str
    input_compiler: str
    dataset: Path
    predictions: Path
    metrics: Path
    prediction_field: str
    sampled_metrics: bool
    protocol: str


def result_path(kind: str, name: str) -> Path:
    return PROJECT / "results" / kind / name


MODELS = (
    ModelSpec(
        "DecIR-1.3B",
        "DecIR-1.3B",
        "Clang",
        CLANG_DATASET,
        result_path("predictions", "decir-1.3b-clang.json"),
        result_path("metrics", "decir-1.3b-clang-legacy.json"),
        "output",
        False,
        "direct greedy generation",
    ),
    ModelSpec(
        "DecIR-6.7B",
        "DecIR-6.7B",
        "Clang",
        CLANG_DATASET,
        result_path("predictions", "decir-6.7b-clang.json"),
        result_path("metrics", "decir-6.7b-clang-legacy.json"),
        "output",
        False,
        "direct greedy generation",
    ),
    ModelSpec(
        "LLM4Decompile-1.3B-v1.5",
        "LLM4Decompile-1.3B-v1.5",
        "GCC",
        GCC_DATASET,
        result_path("predictions", "llm4decompile-1.3b-v1.5-gcc.json"),
        result_path("metrics", "llm4decompile-1.3b-v1.5-gcc-legacy.json"),
        "output",
        False,
        "direct greedy generation",
    ),
    ModelSpec(
        "LLM4Decompile-6.7B-v1.5",
        "LLM4Decompile-6.7B-v1.5",
        "GCC",
        GCC_DATASET,
        result_path("predictions", "llm4decompile-6.7b-v1.5-gcc.json"),
        result_path("metrics", "llm4decompile-6.7b-v1.5-gcc-legacy.json"),
        "output",
        False,
        "direct greedy generation",
    ),
    ModelSpec(
        "sc²dec-6.7B",
        r"sc$^2$dec-6.7B",
        "GCC",
        GCC_DATASET,
        result_path(
            "predictions",
            "sccdec-6.7b-standard-lora-humaneval-gcc.json",
        ),
        result_path(
            "metrics",
            "sccdec-6.7b-standard-lora-humaneval-gcc-final.json",
        ),
        "output",
        False,
        (
            "released FAE LoRA with ordinary alpha/r=2 scaling, one-shot "
            "initial generation, and self-constructed-context second pass"
        ),
    ),
    ModelSpec(
        "SLaDe",
        r"SLaDe$^\dagger$",
        "GCC",
        GCC_DATASET,
        result_path(
            "predictions",
            "slade-released-x86-allopts-humaneval-gcc-typed.json",
        ),
        result_path(
            "metrics",
            "slade-released-x86-allopts-humaneval-gcc-first-beam.json",
        ),
        "typed_candidates",
        True,
        (
            "released O0/O3 checkpoints and first beam; O1/O2 use the O3 "
            "checkpoint as labelled zero-shot transfer"
        ),
    ),
    ModelSpec(
        "Nova-1.3B",
        r"Nova-1.3B$^\ddagger$",
        "GCC",
        GCC_DATASET,
        result_path("predictions", "nova-1.3b-rerun-seed42-gcc.json"),
        result_path(
            "metrics",
            "nova-1.3b-rerun-seed42-gcc-first-sample-legacy.json",
        ),
        "candidates",
        False,
        "official address-aware normalization; first fixed-seed sample",
    ),
    ModelSpec(
        "Nova-6.7B",
        r"Nova-6.7B$^\ddagger$",
        "GCC",
        GCC_DATASET,
        result_path("predictions", "nova-6.7b-seed42-gcc.json"),
        result_path(
            "metrics",
            "nova-6.7b-seed42-gcc-first-sample-legacy.json",
        ),
        "candidates",
        False,
        (
            "official address-aware normalization and hierarchical "
            "attention; first of 20 fixed-seed sampled candidates"
        ),
    ),
    ModelSpec(
        "IDA Pro 9.0 (Hex-Rays)",
        "IDA Pro 9.0 (Hex-Rays)",
        "GCC binary",
        GCC_DATASET,
        result_path(
            "predictions", "ida-pro-9.0-hexrays-gcc.json"
        ),
        result_path(
            "metrics", "ida-pro-9.0-hexrays-gcc-legacy.json"
        ),
        "output",
        False,
        (
            "fresh headless Hex-Rays decompilation of GCC executables; "
            "no debug information or reference types"
        ),
    ),
    ModelSpec(
        "Ghidra 10.4",
        "Ghidra 10.4",
        "GCC binary",
        GCC_DATASET,
        result_path("predictions", "ghidra-10.4-gcc.json"),
        result_path("metrics", "ghidra-10.4-gcc-legacy.json"),
        "output",
        False,
        (
            "fresh headless Ghidra decompilation of GCC executables; "
            "no debug information or reference types"
        ),
    ),
)


def read_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def key(row: dict) -> tuple[str, str]:
    return str(row["task_id"]), str(row["type"])


def validate_dataset(path: Path) -> dict[tuple[str, str], dict]:
    rows = read_rows(path)
    counts = Counter(str(row["type"]) for row in rows)
    expected = Counter({optimization: 164 for optimization in OPTIMIZATIONS})
    if len(rows) != 656 or counts != expected:
        raise ValueError(f"{path} has invalid HumanEval scope: {dict(counts)}")
    by_key = {key(row): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError(f"{path} contains duplicate task/O-level keys")
    return by_key


def candidate(row: dict, field: str) -> str:
    value = row.get(field, "")
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def validate_predictions(
    spec: ModelSpec, references: dict[tuple[str, str], dict]
) -> tuple[list[dict], dict[str, float], float]:
    rows = read_rows(spec.predictions)
    observed = {key(row): row for row in rows}
    if len(rows) != 656 or set(observed) != set(references):
        raise ValueError(
            f"{spec.name} prediction coverage mismatch: "
            f"{len(rows)} rows, {len(observed)} unique keys"
        )
    counts = Counter(str(row["type"]) for row in rows)
    if counts != Counter(
        {optimization: 164 for optimization in OPTIMIZATIONS}
    ):
        raise ValueError(f"{spec.name} has unbalanced predictions")
    if spec.name != "SLaDe":
        mismatched = [
            item_key
            for item_key, row in observed.items()
            if row.get("input_asm_prompt")
            != references[item_key].get("input_asm_prompt")
        ]
        if mismatched:
            raise ValueError(
                f"{spec.name} input assembly differs from {spec.dataset}: "
                f"{mismatched[0]}"
            )
    else:
        transfer = {
            str(row["type"]): str(
                row.get("generation", {}).get("checkpoint_optimization")
            )
            for row in rows
        }
        if transfer != {"O0": "O0", "O1": "O3", "O2": "O3", "O3": "O3"}:
            raise ValueError(f"unexpected SLaDe checkpoint mapping: {transfer}")
        if {
            row.get("generation", {}).get("assembly_source") for row in rows
        } != {"gcc-source"}:
            raise ValueError("SLaDe inputs are not compiler-generated GCC GAS")

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        prediction = candidate(row, spec.prediction_field)
        grouped[str(row["type"])].append(
            100
            * normalized_edit_similarity(
                str(row.get("c_func", "")), prediction
            )
        )
    edit_by_optimization = {
        optimization: round(
            sum(grouped[optimization]) / len(grouped[optimization]), 2
        )
        for optimization in OPTIMIZATIONS
    }
    values = [
        value
        for optimization in OPTIMIZATIONS
        for value in grouped[optimization]
    ]
    return rows, edit_by_optimization, round(sum(values) / len(values), 2)


def direct_metrics(payload: dict) -> tuple[dict, dict]:
    if payload.get("protocol", {}).get("strict") is not False:
        raise ValueError("HumanEval mixed table requires GNU C legacy scoring")
    summary = payload["summary"]
    levels = {
        optimization: {
            "total": summary[optimization]["total"],
            "compile_rate": summary[optimization]["compile_rate"],
            "run_rate": summary[optimization]["run_rate"],
        }
        for optimization in OPTIMIZATIONS
    }
    overall = {
        "total": summary["mean"]["total"],
        "compile_rate": summary["mean"]["compile_rate"],
        "run_rate": summary["mean"]["run_rate"],
    }
    return overall, levels


def sampled_metrics(payload: dict) -> tuple[dict, dict]:
    protocol = payload["protocol"]
    if protocol.get("strict") is not False or not protocol.get(
        "ordered_candidates"
    ):
        raise ValueError("SLaDe metrics must be ordered first-beam GNU C")
    levels = {
        optimization: {
            "total": 164,
            "compile_rate": payload["recompilable"][optimization][
                "first_beam"
            ],
            "run_rate": payload["re_executable"][optimization]["first_beam"],
        }
        for optimization in OPTIMIZATIONS
    }
    overall = {
        "total": 656,
        "compile_rate": payload["recompilable"]["mean"]["first_beam"],
        "run_rate": payload["re_executable"]["mean"]["first_beam"],
    }
    return overall, levels


def summarize(spec: ModelSpec) -> dict:
    references = validate_dataset(spec.dataset)
    _, edit_levels, edit_overall = validate_predictions(spec, references)
    metrics = json.loads(spec.metrics.read_text(encoding="utf-8"))
    overall, levels = (
        sampled_metrics(metrics)
        if spec.sampled_metrics
        else direct_metrics(metrics)
    )
    if overall["total"] != 656 or any(
        levels[optimization]["total"] != 164
        for optimization in OPTIMIZATIONS
    ):
        raise ValueError(f"{spec.name} metric scope is not 164 x 4")
    overall["edit_similarity"] = edit_overall
    for optimization in OPTIMIZATIONS:
        levels[optimization]["edit_similarity"] = edit_levels[optimization]
    return {
        "model": spec.name,
        "input_compiler": spec.input_compiler,
        "instances_per_optimization": 164,
        "metrics": str(spec.metrics.relative_to(PROJECT)),
        "predictions": str(spec.predictions.relative_to(PROJECT)),
        "protocol": spec.protocol,
        "overall": overall,
        "by_optimization": levels,
    }


def pair(values: dict) -> str:
    return f"{values['compile_rate']:.2f}/{values['run_rate']:.2f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "model",
        "input_compiler",
        "instances_per_optimization",
        *(
            f"{optimization}_{metric}"
            for optimization in OPTIMIZATIONS
            for metric in ("compile_rate", "run_rate", "edit_similarity")
        ),
        "average_compile_rate",
        "average_run_rate",
        "average_edit_similarity",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in rows:
            row: dict[str, str | int | float] = {
                "model": result["model"],
                "input_compiler": result["input_compiler"],
                "instances_per_optimization": 164,
            }
            for optimization in OPTIMIZATIONS:
                values = result["by_optimization"][optimization]
                for metric in (
                    "compile_rate",
                    "run_rate",
                    "edit_similarity",
                ):
                    row[f"{optimization}_{metric}"] = values[metric]
            for metric in ("compile_rate", "run_rate", "edit_similarity"):
                row[f"average_{metric}"] = result["overall"][metric]
            writer.writerow(row)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# HumanEval-Decompile C-only mixed-input-compiler comparison",
        "",
        (
            "DecIR uses Clang objdump inputs; neural baselines use GCC "
            "assembly, while IDA and Ghidra use fresh GCC executables. "
            "Every candidate uses the common GCC GNU C execution harness. "
            "Each O-level and average cell is RC/RE (%)."
        ),
        "",
        (
            "| Model | Input compiler | N/level | O0 | O1 | O2 | O3 | "
            "Average | Avg. Edit |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    result["model"],
                    result["input_compiler"],
                    "164",
                    *(
                        pair(result["by_optimization"][optimization])
                        for optimization in OPTIMIZATIONS
                    ),
                    pair(result["overall"]),
                    f"{result['overall']['edit_similarity']:.2f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Protocol notes:",
            "",
            (
                "- sc²dec uses the released FAE LoRA with the paper-consistent "
                "ordinary LoRA scale alpha/r=2 (not the public adapter "
                "metadata's rsLoRA scale), one-shot initial generation, and "
                "the self-constructed-context second pass."
            ),
            (
                "- SLaDe O1/O2 are explicitly zero-shot transfers through "
                "its released O3 checkpoint; O0/O3 use native mappings."
            ),
            "- Nova rows are their first fixed-seed samples, not Pass@k.",
            (
                "- IDA Pro 9.0 uses fresh GCC executables without debug "
                "information or reference type/signature injection."
            ),
            (
                "- Ghidra 10.4 uses fresh GCC executables without debug "
                "information or reference type/signature injection."
            ),
            (
                "- Edit is whitespace-normalized character-level "
                "Levenshtein similarity against the reference C field."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: list[dict]) -> None:
    latex_names = {spec.name: spec.latex_name for spec in MODELS}
    lines = [
        (
            "% Auto-generated by "
            "project/evaluation/humaneval/summarize_standard_mixed.py."
        )
    ]
    for result in rows:
        lines.append(
            " & ".join(
                [
                    latex_names[result["model"]],
                    result["input_compiler"],
                    *(
                        pair(result["by_optimization"][optimization])
                        for optimization in OPTIMIZATIONS
                    ),
                    pair(result["overall"]),
                    f"{result['overall']['edit_similarity']:.2f}",
                ]
            )
            + r" \\"
        )
    lines.append(r"\bottomrule")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=(
            PROJECT
            / "results/metrics/humaneval-c-mixed-compiler-comparison.json"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=(
            PROJECT
            / "results/metrics/humaneval-c-mixed-compiler-comparison.csv"
        ),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=(
            PROJECT
            / "results/metrics/humaneval-c-mixed-compiler-comparison.md"
        ),
    )
    parser.add_argument(
        "--output-latex",
        type=Path,
        default=PAPER / "generated_humaneval_c_mixed_compiler.tex",
    )
    args = parser.parse_args()

    required = [
        path
        for spec in MODELS
        for path in (spec.dataset, spec.predictions, spec.metrics)
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing artifacts: " + ", ".join(missing))
    rows = [summarize(spec) for spec in MODELS]
    payload = {
        "benchmark": "HumanEval-Decompile",
        "language": "C",
        "instances": 656,
        "instances_per_optimization": 164,
        "input_compiler_policy": {
            "DecIR-1.3B": "Clang",
            "DecIR-6.7B": "Clang",
            "neural_baselines": "GCC assembly",
            "IDA Pro 9.0 (Hex-Rays)": "GCC executable",
            "Ghidra 10.4": "GCC executable",
        },
        "candidate_execution_oracle": "GCC GNU C common harness",
        "optimizations": list(OPTIMIZATIONS),
        "results": rows,
    }
    for path in (
        args.output_json,
        args.output_csv,
        args.output_markdown,
        args.output_latex,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_csv, rows)
    write_markdown(args.output_markdown, rows)
    write_latex(args.output_latex, rows)
    print(
        f"Wrote {len(rows)}-model HumanEval summary to {args.output_json}"
    )


if __name__ == "__main__":
    main()
