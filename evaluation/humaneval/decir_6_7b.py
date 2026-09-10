#!/usr/bin/env python3
"""Run DecIR-6.7B inference and fixed-test evaluation."""

from pathlib import Path

from _direct import run_direct


if __name__ == "__main__":
    run_direct(
        model_name="DecIR-6.7B",
        default_model=Path("/home/jiang/model/decir-6.7b"),
        prediction_name="decir-6.7b-humaneval-goron.json",
        metric_name="decir-6.7b-humaneval-goron.json",
    )
