import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from evaluate_predictions import summarize_by_obfuscation  # noqa: E402


def test_obfuscation_summary_is_partitioned():
    rows = [
        {
            "task_id": "humaneval/0/none",
            "type": "O0",
            "obfuscation": "none",
            "compilable": True,
            "re_executable": True,
        },
        {
            "task_id": "humaneval/0/cff",
            "type": "O0",
            "obfuscation": "cff",
            "compilable": True,
            "re_executable": False,
        },
    ]
    summary = summarize_by_obfuscation(rows, bootstrap_samples=20, seed=42)
    assert summary["none"]["mean"]["run_rate"] == 100.0
    assert summary["cff"]["mean"]["run_rate"] == 0.0
