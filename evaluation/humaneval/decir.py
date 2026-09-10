#!/usr/bin/env python3
"""Run DecIR-1.3B inference and fixed-test evaluation."""

from pathlib import Path

from _direct import run_direct


if __name__ == "__main__":
    run_direct(
        model_name="DecIR-1.3B",
        default_model=Path("/home/jiang/model/DecIR-1.3B"),
        prediction_name="decir-1.3b-humaneval-goron.json",
        metric_name="decir-1.3b-humaneval-goron.json",
    )
