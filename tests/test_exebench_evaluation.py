from __future__ import annotations

from evaluation.exebench.common import (
    adapt_candidate_name,
    compile_function_gas,
    diff_io,
    normalize_function_source,
)


def test_normalize_function_source_preserves_local_static_storage() -> None:
    source = "static inline int target(void) { static int calls; return ++calls; }"
    normalized = normalize_function_source(source)

    assert normalized.startswith("  int target")
    assert "{ static int calls;" in normalized


def test_adapt_candidate_name_accepts_only_a_func0_definition() -> None:
    source = "int func0(int value) { return value + 1; }"
    adapted, accepted = adapt_candidate_name(source, "increment")

    assert accepted == "func0"
    assert "int increment(int value)" in adapted


def test_diff_io_recurses_and_uses_numeric_tolerance() -> None:
    observed = {"value": [1, 0.1 + 0.2]}
    expected = {"value": [1, 0.3]}

    assert diff_io(observed, expected)
    assert not diff_io({"value": [1, 0.31]}, expected)


def test_clang_gas_extraction() -> None:
    assembly = compile_function_gas(
        "int func0(int value) { return value + 1; }",
        "func0",
        "O2",
        gcc="clang",
    )
    assert "func0:" in assembly
    assert ".cfi_endproc" in assembly
