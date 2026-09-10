from __future__ import annotations

import json
from pathlib import Path

from evaluation.mbpp.direct_c import expected_shard_ids, prediction_complete
from evaluation.mbpp.prepare_clang import extract_function
from evaluation.mbpp.summarize_c_models import c_summary
from evaluation.run_inference_slade import gcc_source_assembly


def test_expected_shards_keep_optimization_groups_together() -> None:
    rows = [
        {"task_id": f"mbpp/{problem}/{opt}"}
        for problem in range(3)
        for opt in ("O0", "O1", "O2", "O3")
    ]
    assert expected_shard_ids(rows, 0, 2, 4) == [
        *(f"mbpp/0/{opt}" for opt in ("O0", "O1", "O2", "O3")),
        *(f"mbpp/2/{opt}" for opt in ("O0", "O1", "O2", "O3")),
    ]
    assert expected_shard_ids(rows, 1, 2, 4) == [
        *(f"mbpp/1/{opt}" for opt in ("O0", "O1", "O2", "O3")),
    ]


def test_prediction_complete_checks_order_and_model(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    path = tmp_path / "predictions.json"
    rows = [
        {
            "task_id": task_id,
            "output": "int func0(void) { return 0; }",
            "generation": {"model": str(model)},
        }
        for task_id in ("mbpp/0/O0", "mbpp/0/O1")
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert prediction_complete(
        path, ["mbpp/0/O0", "mbpp/0/O1"], model
    )
    assert not prediction_complete(
        path, ["mbpp/0/O1", "mbpp/0/O0"], model
    )
    other_model = tmp_path / "other-model"
    other_model.mkdir()
    assert not prediction_complete(
        path, ["mbpp/0/O0", "mbpp/0/O1"], other_model
    )


def test_c_summary_selects_c_subset_from_mixed_metrics() -> None:
    overall = {"total": 8}
    c_overall = {"total": 4}
    c_levels = {"O0": {"total": 1}}
    payload = {
        "summary": {
            "overall": overall,
            "by_optimization": {"O0": {"total": 2}},
            "by_language": {"c": c_overall, "cpp": {"total": 4}},
            "by_language_and_optimization": {
                "c": c_levels,
                "cpp": {"O0": {"total": 1}},
            },
        },
        "per_case": [{"language": "c"}, {"language": "cpp"}],
    }
    assert c_summary(payload) == (c_overall, c_levels)


def test_slade_can_prepare_o1_and_o2_gas_inputs() -> None:
    source = "int func0(int x) { return x + 1; }"
    for optimization in ("O1", "O2"):
        assembly = gcc_source_assembly(source, optimization, "gcc")
        assert "func0:" in assembly
        assert ".cfi_endproc" in assembly


def test_extract_clang_objdump_function() -> None:
    disassembly = """
0000000000000000 <func0>:
   0:\t8d 47 01             \tlea    0x1(%rdi),%eax
   3:\tc3                   \tret
"""
    assert extract_function(disassembly, "func0") == (
        "func0:\nlea    0x1(%rdi),%eax\nret"
    )


def test_extract_clang_linked_call_symbol() -> None:
    disassembly = """
0000000000001180 <func0>:
    1180:\t55                   \tpush   %rbp
    1181:\te8 aa fe ff ff       \tcall   1030 <strlen@plt>
    1186:\t5d                   \tpop    %rbp
    1187:\tc3                   \tret
"""
    assert extract_function(disassembly, "func0") == (
        "func0:\npush   %rbp\ncall   1030 <strlen@plt>\npop    %rbp\nret"
    )
