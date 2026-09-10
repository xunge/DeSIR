#!/usr/bin/env python3
"""Summarize DecIR results on Clang-generated ExeBench and MBPP inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
OPTIMIZATIONS = ("O0", "O1", "O2", "O3")
MODELS = ("DecIR-1.3B", "DecIR-6.7B")

METRICS = {
    ("ExeBench", "DecIR-1.3B"): (
        PROJECT / "results/metrics/decir-1.3b-exebench-clang-c.json"
    ),
    ("ExeBench", "DecIR-6.7B"): (
        PROJECT / "results/metrics/decir-6.7b-exebench-clang-c.json"
    ),
    ("MBPP-Decompile", "DecIR-1.3B"): (
        PROJECT
        / "results/metrics/decir-1.3b-mbpp-clang-linked-c.json"
    ),
    ("MBPP-Decompile", "DecIR-6.7B"): (
        PROJECT
        / "results/metrics/decir-6.7b-mbpp-clang-linked-c.json"
    ),
}

EXPECTED = {
    "ExeBench": {
        "per_level": 1933,
        "total": 7732,
        "primary_metric": "function_pass_rate",
        "compile_metric": "compile_rate",
        "secondary_metric": "io_accuracy",
    },
    "MBPP-Decompile": {
        "per_level": 973,
        "total": 3892,
        "primary_metric": "run_rate",
        "compile_metric": "compile_rate",
        "secondary_metric": "edit_similarity",
    },
}


def compact_summary(benchmark: str, model: str, path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    overall = summary["overall"]
    levels = summary["by_optimization"]
    expected = EXPECTED[benchmark]
    totals = {opt: levels[opt]["total"] for opt in OPTIMIZATIONS}
    if overall["total"] != expected["total"] or set(totals.values()) != {
        expected["per_level"]
    }:
        raise ValueError(
            f"{path} is not on the expected balanced scope: "
            f"overall={overall['total']}, per-level={totals}"
        )

    selected = (
        expected["compile_metric"],
        expected["primary_metric"],
        expected["secondary_metric"],
    )
    return {
        "benchmark": benchmark,
        "model": model,
        "metrics": str(path.relative_to(PROJECT)),
        "instances": expected["total"],
        "instances_per_optimization": expected["per_level"],
        "primary_metric": expected["primary_metric"],
        "overall": {key: overall[key] for key in selected},
        "by_optimization": {
            opt: {key: levels[opt][key] for key in selected}
            for opt in OPTIMIZATIONS
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "benchmark",
        "model",
        "optimization",
        "instances",
        "compile_rate",
        "primary_metric",
        "primary_rate",
        "secondary_metric",
        "secondary_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            secondary = EXPECTED[row["benchmark"]]["secondary_metric"]
            for opt in (*OPTIMIZATIONS, "Average"):
                values = (
                    row["overall"]
                    if opt == "Average"
                    else row["by_optimization"][opt]
                )
                writer.writerow(
                    {
                        "benchmark": row["benchmark"],
                        "model": row["model"],
                        "optimization": opt,
                        "instances": (
                            row["instances"]
                            if opt == "Average"
                            else row["instances_per_optimization"]
                        ),
                        "compile_rate": values["compile_rate"],
                        "primary_metric": row["primary_metric"],
                        "primary_rate": values[row["primary_metric"]],
                        "secondary_metric": secondary,
                        "secondary_rate": values[secondary],
                    }
                )


def pair(row: dict, values: dict) -> str:
    return (
        f"{values['compile_rate']:.2f}/"
        f"{values[row['primary_metric']]:.2f}"
    )


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# DecIR on Clang-generated C-only inputs",
        "",
        (
            "Clang 18.1.3 generated every model input at O0--O3. "
            "MBPP inputs are linked before disassembly so call symbols "
            "are resolved. "
            "Generated candidates use the common GCC 13.3.0 execution oracle "
            "to isolate the input-compiler change."
        ),
        "",
    ]
    for benchmark in ("ExeBench", "MBPP-Decompile"):
        selected = [row for row in rows if row["benchmark"] == benchmark]
        if benchmark == "ExeBench":
            note = (
                "Each cell is compile/function pass (%); a function passes "
                "only when all 10 official real I/O tests pass."
            )
            secondary_label = "Avg. I/O"
        else:
            note = "Each cell is RC/RE (%)."
            secondary_label = "Avg. Edit"
        secondary = EXPECTED[benchmark]["secondary_metric"]
        lines.extend(
            [
                f"## {benchmark}",
                "",
                note,
                "",
                (
                    "| Model | O0 | O1 | O2 | O3 | Average | "
                    f"{secondary_label} |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in selected:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["model"],
                        *(
                            pair(row, row["by_optimization"][opt])
                            for opt in OPTIMIZATIONS
                        ),
                        pair(row, row["overall"]),
                        f"{row['overall'][secondary]:.2f}",
                    ]
                )
                + " |"
            )
        lines.extend(["",])

    lines.extend(
        [
            "Scope:",
            "",
            "- ExeBench: 1,933 validated C functions, 7,732 rows.",
            "- MBPP-Decompile: 973 C problems, 3,892 rows.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: list[dict]) -> None:
    lines = [
        "% Auto-generated by project/evaluation/summarize_clang_decir.py.",
    ]
    benchmark_names = {
        "ExeBench": r"ExeBench (Comp./Func.)",
        "MBPP-Decompile": r"MBPP (RC/RE)",
    }
    for row in rows:
        lines.append(
            " & ".join(
                [
                    benchmark_names[row["benchmark"]],
                    row["model"],
                    *(
                        pair(row, row["by_optimization"][opt])
                        for opt in OPTIMIZATIONS
                    ),
                    pair(row, row["overall"]),
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
        default=PROJECT / "results/metrics/decir-clang-c-comparison.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT / "results/metrics/decir-clang-c-comparison.csv",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=PROJECT / "results/metrics/decir-clang-c-comparison.md",
    )
    parser.add_argument(
        "--output-latex",
        type=Path,
        default=PROJECT.parent / "paper/generated_decir_clang_c.tex",
    )
    args = parser.parse_args()

    missing = [str(path) for path in METRICS.values() if not path.is_file()]
    if missing:
        parser.error("missing metric artifacts: " + ", ".join(missing))

    rows = [
        compact_summary(benchmark, model, METRICS[(benchmark, model)])
        for benchmark in ("ExeBench", "MBPP-Decompile")
        for model in MODELS
    ]
    payload = {
        "input_compiler": "clang version 18.1.3",
        "candidate_execution_compiler": (
            "gcc/g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
        ),
        "language": "C",
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
    print(f"Wrote Clang-input comparison to {args.output_json}")


if __name__ == "__main__":
    main()
