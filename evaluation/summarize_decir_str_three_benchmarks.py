#!/usr/bin/env python3
"""Create a three-benchmark DecIR-STR RC/RE/Edit/TCP report."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import editdistance


OPTS = ("O0", "O1", "O2", "O3")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def edit_similarity(target: str, prediction: str) -> float:
    target = "\n".join(line.strip() for line in target.splitlines() if line.strip())
    prediction = "\n".join(
        line.strip() for line in prediction.splitlines() if line.strip()
    )
    denominator = max(len(target), len(prediction))
    return 100.0 if not denominator else 100 * (
        1 - editdistance.eval(target, prediction) / denominator
    )


def exebench_edits(predictions: list[dict]) -> dict[str, float]:
    grouped = {
        opt: [
            edit_similarity(str(row.get("c_func", "")), str(row["output"]))
            for row in predictions
            if row["type"] == opt
        ]
        for opt in OPTS
    }
    return {
        **{opt: sum(values) / len(values) for opt, values in grouped.items()},
        "Average": sum(sum(values) for values in grouped.values())
        / sum(len(values) for values in grouped.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--humaneval-summary", type=Path, required=True)
    parser.add_argument("--mbpp-metrics", type=Path, required=True)
    parser.add_argument("--mbpp-tcp", type=Path, required=True)
    parser.add_argument("--exebench-metrics", type=Path, required=True)
    parser.add_argument("--exebench-predictions", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    human = read(args.humaneval_summary)
    mbpp = read(args.mbpp_metrics)["summary"]
    mbpp_tcp = read(args.mbpp_tcp)["summary"]
    exe = read(args.exebench_metrics)["summary"]
    exe_predictions = read(args.exebench_predictions)
    exe_edit = exebench_edits(exe_predictions)
    human_by_opt = {row["optimization"]: row for row in human["by_optimization"]}

    rows = []
    for benchmark in (
        "HumanEval-Decompile",
        "MBPP-Decompile C-only",
        "ExeBench C-only",
    ):
        for opt in (*OPTS, "Average"):
            if benchmark == "HumanEval-Decompile":
                group = human["average"] if opt == "Average" else human_by_opt[opt]
                values = (group["rc"], group["re"], group["edit"], group["tcp"])
                total = 656 if opt == "Average" else group["functions"]
            elif benchmark.startswith("MBPP"):
                group = mbpp["overall"] if opt == "Average" else mbpp["by_optimization"][opt]
                tcp_group = mbpp_tcp["overall"] if opt == "Average" else mbpp_tcp["by_optimization"][opt]
                values = (
                    group["compile_rate"], group["run_rate"],
                    group["edit_similarity"], tcp_group["tcp"],
                )
                total = group["total"]
            else:
                group = exe["overall"] if opt == "Average" else exe["by_optimization"][opt]
                values = (
                    group["compile_rate"], group["function_pass_rate"],
                    exe_edit[opt], group["io_accuracy"],
                )
                total = group["total"]
            rows.append({
                "benchmark": benchmark, "optimization": opt, "functions": total,
                "RC": round(values[0], 2), "RE": round(values[1], 2),
                "Edit": round(values[2], 2), "TCP": round(values[3], 2),
            })

    payload = {
        "model": args.model_name,
        "model_path": str(Path(args.model_path).resolve()),
        "input_compiler": "Clang 18.1.3",
        "candidate_execution_compiler": "GCC/G++ 13.3.0",
        "rows": rows,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {ext: Path(str(args.output_prefix) + f".{ext}") for ext in ("json", "csv", "md")}
    temporary = Path(str(paths["json"]) + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, paths["json"])
    with paths["csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {args.model_name}: Clang results", "",
        "| Benchmark | Opt | N | RC | RE | Edit | TCP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['benchmark']} | {row['optimization']} | {row['functions']} "
        f"| {row['RC']:.2f} | {row['RE']:.2f} | {row['Edit']:.2f} | {row['TCP']:.2f} |"
        for row in rows
    )
    paths["md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("Wrote " + ", ".join(str(path) for path in paths.values()))


if __name__ == "__main__":
    main()
