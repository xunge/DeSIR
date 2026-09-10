from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset"))
sys.path.insert(0, str(ROOT / "evaluation"))

from common import compile_and_test  # noqa: E402
from compare_decompile_bench import paired_interval  # noqa: E402
from differential_test import evaluate as differential_evaluate  # noqa: E402
from evaluate_decompile_bench import (  # noqa: E402
    evaluate_one as evaluate_decompile_bench_one,
    init_worker as init_decompile_bench_worker,
    summarize as summarize_decompile_bench,
)
from import_decompile_bench import convert  # noqa: E402
from ir_normalization import normalize_ir_legacy, normalize_ir_safe  # noqa: E402


RAW_IR = r"""
; ModuleID = 'sample.c'
source_filename = "sample.c"
target triple = "x86_64-pc-linux-gnu"
define dso_local i32 @helper(i32 noundef %0) #0 !dbg !10 {
  %2 = add nsw i32 %0, 1, !dbg !11
  ret i32 %2, !dbg !12
}
define dso_local i32 @target(i32 noundef %0) #0 !dbg !20 {
  %2 = call i32 @helper(i32 noundef %0), !dbg !21, !range !30
  call void @side(), !dbg !21
  ret i32 %2, !dbg !22
}
declare void @side()
declare void @llvm.dbg.value(metadata, metadata, metadata) #1
attributes #0 = { nounwind }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
!10 = distinct !DISubprogram(name: "helper")
!11 = !DILocation(line: 1, column: 1, scope: !10)
!12 = !DILocation(line: 1, column: 2, scope: !10)
!20 = distinct !DISubprogram(name: "target")
!21 = !DILocation(line: 2, column: 1, scope: !20)
!22 = !DILocation(line: 2, column: 2, scope: !20)
!30 = !{i32 0, i32 10}
"""


class NormalizationTests(unittest.TestCase):
    def test_safe_preserves_semantic_syntax(self) -> None:
        normalized = normalize_ir_safe(RAW_IR)
        self.assertIn("call i32 @helper", normalized)
        self.assertIn("call void @side", normalized)
        self.assertIn("declare void @side", normalized)
        self.assertIn("!range !30", normalized)
        self.assertIn("!30 = !{i32 0, i32 10}", normalized)
        self.assertIn("add nsw", normalized)
        self.assertIn("attributes #0", normalized)
        self.assertNotIn("!dbg", normalized)
        self.assertNotIn("@llvm.dbg", normalized)

    def test_legacy_reproduces_lossy_call_removal(self) -> None:
        normalized = normalize_ir_legacy(RAW_IR)
        self.assertIn("call i32 @helper", normalized)
        self.assertNotIn("call void @side", normalized)
        self.assertNotIn("declare void @side", normalized)
        self.assertNotIn("attributes #0", normalized)


class EvaluationTests(unittest.TestCase):
    def test_compile_and_fixed_test(self) -> None:
        result = compile_and_test(
            "int square(int x) { return x * x; }",
            "int main(void) { return square(4) == 16 ? 0 : 1; }",
            "int square(int x) { return x * x; }",
        )
        self.assertTrue(result.compilable)
        self.assertTrue(result.re_executable)

    def test_compile_and_test_replaces_non_utf8_runtime_stderr(self) -> None:
        result = compile_and_test(
            "int func0(void) { return 0; }",
            (
                "int main(void) { unsigned char c = 0x80; "
                "fwrite(&c, 1, 1, stderr); return 1; }"
            ),
            "int func0(void) { return 0; }",
            prefix="#include <stdio.h>",
            strict=False,
        )
        self.assertTrue(result.compilable)
        self.assertFalse(result.re_executable)
        self.assertIn("\ufffd", result.stderr)

    def test_compile_and_test_isolates_runtime_working_directory(self) -> None:
        marker = ROOT / "decir-runtime-side-effect.tmp"
        marker.unlink(missing_ok=True)
        try:
            result = compile_and_test(
                "int func0(void) { return 0; }",
                (
                    "int main(void) { FILE *f = fopen("
                    '"decir-runtime-side-effect.tmp", "w"); '
                    "if (!f) return 1; fclose(f); return 0; }"
                ),
                "int func0(void) { return 0; }",
                prefix="#include <stdio.h>",
                strict=False,
            )
            self.assertTrue(result.re_executable)
            self.assertFalse(marker.exists())
        finally:
            marker.unlink(missing_ok=True)

    def test_differential_scalar_subset(self) -> None:
        result = differential_evaluate(
            {
                "task_id": "synthetic/0",
                "function_name": "square",
                "c_func": "int square(int x) { return x * x; }",
                "output": "int square(int x) { return x * x; }",
            },
            compiler="clang",
            trials=100,
            seed=42,
            timeout=10,
        )
        self.assertTrue(result["eligible"])
        self.assertTrue(result["compiled"])
        self.assertTrue(result["passed"])

    def test_decompile_bench_language_optimization_summary(self) -> None:
        rows = [
            {
                "task_id": f"synthetic/{index}/{optimization}",
                "cluster_id": f"synthetic/{index}",
                "language": language,
                "type": optimization,
                "compilable": True,
                "re_executable": passed,
                "edit_similarity": 0.5,
            }
            for index, (language, optimization, passed) in enumerate(
                [
                    ("c", "O0", True),
                    ("c", "O1", False),
                    ("cpp", "O0", False),
                    ("cpp", "O1", True),
                ]
            )
        ]
        summary = summarize_decompile_bench(rows, bootstrap_samples=10, seed=42)
        self.assertEqual(
            summary["by_language_and_optimization"]["c"]["O0"]["run_rate"],
            100.0,
        )
        self.assertEqual(
            summary["by_language_and_optimization"]["cpp"]["O0"]["run_rate"],
            0.0,
        )

    def test_paired_cluster_interval_direction(self) -> None:
        pairs = [
            (
                {
                    "task_id": f"synthetic/{index}/O0",
                    "cluster_id": f"synthetic/{index}",
                    "re_executable": True,
                },
                {
                    "task_id": f"synthetic/{index}/O0",
                    "cluster_id": f"synthetic/{index}",
                    "re_executable": False,
                },
            )
            for index in range(4)
        ]
        point, interval = paired_interval(
            pairs,
            "re_executable",
            samples=100,
            seed=42,
        )
        self.assertEqual(point, 100.0)
        self.assertEqual(interval, [100.0, 100.0])

    def test_decompile_bench_replaces_non_utf8_runtime_stderr(self) -> None:
        init_decompile_bench_worker(
            {
                "prediction_field": "output",
                "c_compiler": "gcc",
                "cpp_compiler": "g++",
                "crypto_library": "-lcrypto",
                "optimization": "-O0",
                "timeout": 10,
            }
        )
        result = evaluate_decompile_bench_one(
            {
                "task_id": "synthetic/non-utf8",
                "cluster_id": "synthetic/non-utf8",
                "language": "c",
                "type": "O0",
                "c_prefix": "#include <stdio.h>",
                "c_func": "int func0(void) { return 0; }",
                "output": "int func0(void) { return 0; }",
                "c_test": (
                    "int main(void) { unsigned char c = 0x80; "
                    "fwrite(&c, 1, 1, stderr); return 1; }"
                ),
            }
        )
        self.assertTrue(result["compilable"])
        self.assertFalse(result["re_executable"])
        self.assertIn("\ufffd", result["stderr"])


class DecompileBenchImportTests(unittest.TestCase):
    def test_groups_optimization_variants_by_source_program(self) -> None:
        rows = []
        for index, opt in enumerate(("O0", "O1", "O2", "O3")):
            rows.append(
                {
                    "index": index,
                    "func_name": "func0",
                    "func_dep": "#include <assert.h>",
                    "func": "int func0(int x) { return x + 1; }",
                    "test": "int main(void) { assert(func0(1) == 2); }",
                    "opt": opt,
                    "language": "c",
                    "asm": f"func0:\n# {opt}",
                }
            )
        converted = convert(rows, "mbpp-decompile")
        self.assertEqual(len(converted), 4)
        self.assertEqual(
            {row["task_id"].rsplit("/", 1)[0] for row in converted},
            {"mbpp-decompile/0000"},
        )
        self.assertEqual(
            {row["cluster_id"] for row in converted},
            {"mbpp-decompile/0000"},
        )
        self.assertEqual({row["compiler"] for row in converted}, {"gcc"})

    def test_clusters_parallel_c_and_cpp_translations(self) -> None:
        rows = []
        for language in ("c", "cpp"):
            for opt in ("O0", "O1", "O2", "O3"):
                rows.append(
                    {
                        "index": len(rows),
                        "func_name": "func0",
                        "func_dep": "#include <assert.h>",
                        "func": (
                            "int func0(int x) { return x + 1; }"
                            if language == "c"
                            else "int func0(int x) { return x + 1; }"
                        ),
                        "test": "int main() { assert(func0(1) == 2); }",
                        "opt": opt,
                        "language": language,
                        "asm": f"func0:\n# {language} {opt}",
                    }
                )
        converted = convert(rows, "mbpp-decompile")
        self.assertEqual(len(converted), 8)
        self.assertEqual(
            {row["cluster_id"] for row in converted},
            {"mbpp-decompile/0000"},
        )
        self.assertEqual(
            {row["task_id"].rsplit("/", 1)[0] for row in converted},
            {"mbpp-decompile/0000", "mbpp-decompile/0001"},
        )
        self.assertEqual(
            {row["compiler"] for row in converted},
            {"gcc", "g++"},
        )


if __name__ == "__main__":
    unittest.main()
