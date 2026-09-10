#!/usr/bin/env python3
"""Create paper-ready MBPP/ExeBench RC, RE, Edit, and TCP tables."""

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


def exe_edits(predictions: list[dict]) -> dict[str, float]:
    result = {}
    for opt in OPTS:
        values = [
            edit_similarity(str(row.get("c_func", "")), str(row["output"]))
            for row in predictions
            if row["type"] == opt
        ]
        result[opt] = sum(values) / len(values)
    result["Average"] = sum(
        edit_similarity(str(row.get("c_func", "")), str(row["output"]))
        for row in predictions
    ) / len(predictions)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mbpp-metrics", type=Path, required=True)
    parser.add_argument("--mbpp-tcp", type=Path, required=True)
    parser.add_argument("--exebench-metrics", type=Path, required=True)
    parser.add_argument("--exebench-predictions", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    mbpp = read(args.mbpp_metrics)["summary"]
    mbpp_tcp = read(args.mbpp_tcp)["summary"]
    exe = read(args.exebench_metrics)["summary"]
    exe_predictions = read(args.exebench_predictions)
    edits = exe_edits(exe_predictions)

    rows = []
    for benchmark in ("MBPP-Decompile C-only", "ExeBench C-only"):
        for opt in (*OPTS, "Average"):
            if benchmark.startswith("MBPP"):
                group = mbpp["overall"] if opt == "Average" else mbpp["by_optimization"][opt]
                tcp_group = (
                    mbpp_tcp["overall"]
                    if opt == "Average"
                    else mbpp_tcp["by_optimization"][opt]
                )
                rc, re_value = group["compile_rate"], group["run_rate"]
                edit, tcp = group["edit_similarity"], tcp_group["tcp"]
                total = group["total"]
            else:
                group = exe["overall"] if opt == "Average" else exe["by_optimization"][opt]
                rc, re_value = group["compile_rate"], group["function_pass_rate"]
                edit, tcp = edits[opt], group["io_accuracy"]
                total = group["total"]
            rows.append(
                {
                    "benchmark": benchmark,
                    "optimization": opt,
                    "functions": total,
                    "RC": round(rc, 2),
                    "RE": round(re_value, 2),
                    "Edit": round(edit, 2),
                    "TCP": round(tcp, 2),
                }
            )

    payload = {
        "model": "/home/jiang/model/decir-6.7b-str",
        "input_compiler": "Clang",
        "candidate_execution_compiler": "GCC/G++ (benchmark oracle)",
        "metric_definitions": {
            "RC": "percentage of generated functions that compile",
            "RE": "percentage of functions passing all tests",
            "Edit": "mean normalized character edit similarity (%)",
            "TCP": "macro mean per-function test-case pass rate (%)",
        },
        "rows": rows,
        "artifacts": {
            "mbpp_metrics": str(args.mbpp_metrics),
            "mbpp_tcp": str(args.mbpp_tcp),
            "exebench_metrics": str(args.exebench_metrics),
            "exebench_predictions": str(args.exebench_predictions),
        },
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    # The model name contains dots (``6.7b``), so with_suffix() would
    # incorrectly truncate most of the requested artifact stem.
    json_path = Path(str(args.output_prefix) + ".json")
    csv_path = Path(str(args.output_prefix) + ".csv")
    md_path = Path(str(args.output_prefix) + ".md")
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, json_path)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# DecIR-6.7B-STR: Clang C-only results",
        "",
        "| Benchmark | Opt | N | RC | RE | Edit | TCP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['benchmark']} | {row['optimization']} | {row['functions']} "
        f"| {row['RC']:.2f} | {row['RE']:.2f} | {row['Edit']:.2f} "
        f"| {row['TCP']:.2f} |"
        for row in rows
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {json_path}, {csv_path}, and {md_path}")


if __name__ == "__main__":
    main()
