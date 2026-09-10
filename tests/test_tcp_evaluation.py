from evaluation.evaluate_tcp import instrument_tests, run_source, summarize


def _tcp_row(
    optimization: str,
    obfuscation: str,
    tcp: float,
) -> dict:
    return {
        "type": optimization,
        "obfuscation": obfuscation,
        "tcp": tcp,
        "all_tests_pass": tcp == 1.0,
        "compilable": True,
        "tests_passed": int(tcp * 10),
        "tests_total": 10,
        "test_count_matches_reference": True,
        "timed_out": False,
    }


def test_summary_groups_goron_obfuscations() -> None:
    rows = [
        _tcp_row("O0", "none", 1.0),
        _tcp_row("O3", "none", 0.5),
        _tcp_row("O0", "cff", 0.0),
        _tcp_row("O3", "cff", 0.25),
    ]
    summary = summarize(rows)
    assert summary["overall"]["tcp"] == 43.75
    assert summary["by_obfuscation"]["none"]["tcp"] == 75.0
    assert summary["by_obfuscation"]["cff"]["tcp"] == 12.5
    assert set(summary["by_optimization"]) == {"O0", "O3"}


def test_tcp_records_partial_passes_without_aborting() -> None:
    item = {
        "c_func": "int func0(int value) { return value + 1; }",
        "c_prefix": "",
        "c_test": """
#include <assert.h>
int main(void) {
    int observed = func0(1);
    assert(observed == 2);
    observed = func0(2);
    assert(observed == 4);
    observed = func0(3);
    assert(observed == 4);
    return 0;
}
""",
    }
    result, kind, sites = run_source(
        item,
        item["c_func"],
        compiler="gcc",
        compile_timeout=10,
        run_timeout=10,
    )
    assert kind == "assert"
    assert sites == 3
    assert result.compilable
    assert result.run_returncode == 0
    assert result.tests_passed == 2
    assert result.tests_observed == 3


def test_tcp_recovers_ignored_boolean_comparator_calls() -> None:
    source = """
int issame(int left, int right) { return left == right; }
int main(void) {
    issame(func0(1), 2);
    issame(func0(2), 3);
    return 0;
}
"""
    rewritten, kind, sites = instrument_tests(source)
    assert kind == "issame"
    assert sites == 2
    assert rewritten.count("TCP_CHECK_EXPR") == 2


def test_tcp_rejects_harness_without_checkable_oracle() -> None:
    source = "int main(void) { func0(1); return 0; }"
    rewritten, kind, sites = instrument_tests(source)
    assert rewritten == source
    assert kind == "no_checkable_oracle"
    assert sites == 0
