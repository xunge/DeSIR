# HumanEval-Decompile evaluation

## DecIR-6.7B-STR（Clang 字符串感知输入）

`decir_str.py` 复用 `evaluation_string/decompile_binary.py` 的输入协议：
Clang 18.1.3 将每个参考函数编译成共享库，从二进制恢复目标函数引用的
字符串并在汇编中标记为 `STR_i`，模型以 temperature 0 生成首候选，
之后按函数的恢复表确定性展开 `STR_i`。与附件的批量评估口径一致，候选
C 代码统一由 GCC 13.3.0 重新编译并执行。

从 `/home/jiang/papers/DecIR/project` 运行：

```bash
/home/jiang/miniconda3/envs/pt271/bin/python \
  evaluation/humaneval/decir_str.py \
  --model /home/jiang/model/decir-6.7b-str \
  --gpu 0 --workers 32 --prompt-workers 16
```

已有完整预测时，仅重新计算指标：

```bash
/home/jiang/miniconda3/envs/pt271/bin/python \
  evaluation/humaneval/decir_str.py --score-only
```

本次 656 个样本的结果如下（均为百分比）：

| Optimization | RC | RE | Edit | TCP |
|---|---:|---:|---:|---:|
| O0 | 92.68 | 83.54 | 45.12 | 85.41 |
| O1 | 92.68 | 60.98 | 37.05 | 70.74 |
| O2 | 92.07 | 59.76 | 36.93 | 68.27 |
| O3 | 93.29 | 60.37 | 37.10 | 68.73 |
| Average | 92.68 | 66.16 | 39.05 | 73.29 |

TCP 按函数先计算“通过测试数/总测试数”再宏平均。四个优化级别中各有
一个来自同一原始函数的无效参考测试 oracle，因此 TCP 使用 652 个有效
函数（每级 163 个）；RC、RE 和 Edit 使用全部 656 个预测。由于模型生成
的 C 可能包含未定义行为，TCP 重复执行三次并选择总体 TCP 的中位数运行；
三次总体值为 73.29、73.28、73.29。

输出为：

- `decompile-eval/humaneval-decompile-clang-string-aware.json`
- `results/predictions/decir-6.7b-str-humaneval-clang.json`
- `results/metrics/decir-6.7b-str-humaneval-clang.json`
- `results/metrics/tcp/decir-6.7b-str-humaneval-clang.json`
- `results/metrics/decir-6.7b-str-humaneval-clang-summary.{json,md}`

## IDA Pro 9.0 / Hex-Rays on fresh GCC binaries

The IDA baseline compiles all 656 HumanEval-Decompile rows (164 C functions
at each of O0--O3) with the benchmark's original executable command, then
invokes the user-supplied IDA at `/home/jiang/ida-pro-9.0/idat` in headless
mode. The binaries contain symbols but no debug information. Reference
function signatures and source types are not applied to IDA.

Run from `/home/jiang/papers/DecIR/project`:

```bash
python evaluation/humaneval/ida_gcc.py --workers 8

python evaluation/evaluate_predictions.py \
  --predictions results/predictions/ida-pro-9.0-hexrays-gcc.json \
  --prediction-field output \
  --output results/metrics/ida-pro-9.0-hexrays-gcc-legacy.json \
  --compiler gcc --workers 32 --no-strict \
  --bootstrap-samples 10000 --seed 42

python evaluation/humaneval/summarize_ida_gcc.py
```

`ida_gcc.py` checkpoints each completed case and resumes by default. Use
`--force` to regenerate all pseudocode. Its only postprocessing is removal of
IDA comments/calling-convention annotations, primitive IDA type-alias
normalization, hexadecimal-literal normalization, `clang-format`, and
standalone declarations for globals referenced by Hex-Rays' C tree. Global
sizes and bytes are read from IDA's binary image, not from reference source.
Compatibility macros used by the execution harness are stored separately in
`c_prefix` and therefore do not inflate Edit similarity.

Outputs:

- `results/predictions/ida-pro-9.0-hexrays-gcc.json`: raw and normalized
  pseudocode for every case.
- `results/metrics/ida-pro-9.0-hexrays-gcc-legacy.json`: per-case RC/RE and
  confidence intervals.
- `results/metrics/ida-pro-9.0-hexrays-gcc-summary.{json,csv,md}`: O0--O3,
  averages, and Edit similarity.

## Ghidra 10.4 on fresh GCC binaries

The Ghidra baseline uses the same 656-case GCC executable scope as the IDA
run. It invokes the user-supplied
`/home/jiang/ghidra_10.4_PUBLIC/support/analyzeHeadless` with a matching
Temurin 17 runtime at `/home/jiang/tools/temurin-17`. The analyzed binaries
contain normal ELF symbols but no debug information, and the exporter does
not apply reference signatures or source types.

Run from `/home/jiang/papers/DecIR/project`:

```bash
python evaluation/humaneval/ghidra_gcc.py --workers 8

python evaluation/evaluate_predictions.py \
  --predictions results/predictions/ghidra-10.4-gcc.json \
  --prediction-field output \
  --output results/metrics/ghidra-10.4-gcc-legacy.json \
  --compiler gcc --workers 32 --no-strict \
  --bootstrap-samples 10000 --seed 42

python evaluation/humaneval/summarize_ghidra_gcc.py
```

`ghidra_gcc.py` checkpoints after every completed case and resumes by
default; `--force` reruns the selected scope. `--renormalize` refreshes
`output` and `c_prefix` from saved raw pseudocode without launching Ghidra.
Its postprocessing removes comments, maps Ghidra primitive types, helper
intrinsics, and exact-width partial-scalar lvalues into GNU C, formats the
function, and emits declarations for globals actually referenced by the
pseudocode. Global sizes and initial bytes come from Ghidra's memory image,
never from the reference source. Compatibility declarations are kept in
`c_prefix`, outside the Edit-similarity candidate.

Outputs:

- `results/predictions/ghidra-10.4-gcc.json`: raw and normalized pseudocode,
  generation metadata, and binary-derived global records for every case.
- `results/metrics/ghidra-10.4-gcc-legacy.json`: per-case RC/RE and confidence
  intervals.
- `results/metrics/ghidra-10.4-gcc-summary.{json,csv,md}`: O0--O3, averages,
  and Edit similarity.

The completed run obtains O0/O1/O2/O3 RC/RE of 98.17/76.83,
96.34/67.68, 95.73/64.02, and 89.63/61.59, respectively. The macro average
is 94.97/67.53 and average Edit is 22.53 (all values are percentages).

## Standard C-only O0--O3 mixed-input-compiler comparison

The standard benchmark contains 164 C problems at each of O0, O1, O2, and
O3. DecIR consumes the Clang objdump dataset; LLM4Decompile, sc²dec, SLaDe,
and both Nova sizes consume GCC assembly inputs; IDA and Ghidra consume fresh GCC
executables. Candidate C always uses the common GCC GNU C execution harness.

Run or reuse the two 6.7B direct checkpoints:

```bash
python evaluation/humaneval/standard_direct.py \
  --model /home/jiang/model/decir-6.7b \
  --dataset decompile-eval/decompile-eval-executable-clang-obj.json \
  --predictions results/predictions/decir-6.7b-clang.json \
  --metrics results/metrics/decir-6.7b-clang-legacy.json --gpu 0

python evaluation/humaneval/standard_direct.py \
  --model /home/jiang/model/llm4decompile-6.7b-v1.5 \
  --dataset decompile-eval/decompile-eval-executable-gcc-obj.json \
  --predictions results/predictions/llm4decompile-6.7b-v1.5-gcc.json \
  --metrics results/metrics/llm4decompile-6.7b-v1.5-gcc-legacy.json --gpu 1
```

Run SLaDe at all four levels. O0/O3 retain their released checkpoint mapping;
O1/O2 are explicitly routed through the released O3 checkpoint as zero-shot
transfer:

```bash
python evaluation/humaneval/standard_slade.py --gpu 0 --workers 32
```

Reproduce Nova-6.7B with the official 20-sample setting on two GPUs. The
runner resumes from per-item JSONL checkpoints by default:

```bash
python evaluation/humaneval/nova_6_7b.py
python evaluation/humaneval/summarize_nova_6_7b.py
```

The fresh first candidate obtains O0/O1/O2/O3 RC/RE of
93.29/49.39, 90.24/31.71, 87.20/31.10, and 85.37/25.00. Its average
RC/RE/Edit is 89.02/34.30/47.52. Across all 20 candidates, RC
Pass@1/Pass@10 is 88.69/94.35 and RE Pass@1/Pass@10 is 34.44/48.22.

Regenerate Nova-1.3B's uniform first-sample score and the ten-model summary:

```bash
python evaluation/evaluate_predictions.py \
  --predictions results/predictions/nova-1.3b-rerun-seed42-gcc.json \
  --prediction-field output \
  --output \
    results/metrics/nova-1.3b-rerun-seed42-gcc-first-sample-legacy.json \
  --compiler gcc --workers 32 --no-strict \
  --bootstrap-samples 10000 --seed 42

python evaluation/humaneval/summarize_standard_mixed.py
```

The final table is
`results/metrics/humaneval-c-mixed-compiler-comparison.{json,csv,md}`, and
the paper rows are written to
`../paper/generated_humaneval_c_mixed_compiler.tex`.

## Goron-obfuscated evaluation

本目录是 Goron 混淆实验的可执行入口。所有命令均从
`/home/jiang/papers/DecIR/project` 运行。数据集包含同一 Goron Clang
生成的未混淆对照，以及分别启用 INDBR、ICALL、INDGV、CSE、CFF 的
五组样本；五个 pass 不叠加。

Goron 的 `-mllvm -irobf-*` 是 LLVM/Clang pass，不能由 GCC 生成。
因此所有方法必须接收同一批 Goron-Clang 混淆输入，才能进行受控比较；
所有候选代码再统一使用 GCC 13.3.0 编译和执行。这里的“GCC baseline”
表示统一的候选执行 oracle，并不表示用 GCC 生成一个不存在的 Goron
混淆二进制。

## 独立运行

```bash
python evaluation/humaneval/prepare.py
python evaluation/humaneval/decir.py --gpu 0
python evaluation/humaneval/decir_6_7b.py --gpu 0
python evaluation/humaneval/llm4decompile.py --gpu 1
python evaluation/humaneval/llm4decompile_6_7b.py --gpu 1
python evaluation/humaneval/sccdec.py --gpu 0
python evaluation/humaneval/slade.py --gpu 1
python evaluation/humaneval/nova.py --gpu-a 1 --gpu-b 0
python evaluation/humaneval/nova_goron_6_7b.py --gpu-a 0 --gpu-b 1
python evaluation/humaneval/aggregate.py
```

每个入口同时负责推理与评分；如果完整预测文件已存在，会复用预测并
重新评分。使用 `--force-inference` 可以重做对应模型的推理。Nova 支持
断点续跑，分成两个 GPU lane。Nova-1.3B 保留论文的 20-sample 设置；
本次 Nova-6.7B 表格仅使用一个固定批次 seed 的样本（input batch 8、
temperature 0.2、top-p 0.95），避免为只统计单样本 RC/RE/Edit/TCP
额外生成 19 个未使用候选。两者输出上限均为 1024 token。

DecIR-6.7B 入口默认加载实际本地目录
`/home/jiang/model/decir-6.7b`（末尾为小写 `b`）。本次完整运行的总体
RC/RE 为 64.05/31.43；分配置结果与其他模型一起保存在最终汇总中。

统一运行：

```bash
python evaluation/humaneval/run_all.py
```

若 Nova 的分片预测已经存在，可只合并、评分并重新聚合：

```bash
python evaluation/humaneval/nova.py --score-only
python evaluation/humaneval/nova_goron_6_7b.py --score-only
python evaluation/humaneval/aggregate.py
```

## 输出

- 原始预测：`results/predictions/*humaneval-goron*.json`
- 每样本及分混淆指标：`results/metrics/*humaneval-goron*.json`
- 最终机器可读汇总：`results/metrics/humaneval-goron-summary.json`
- 最终 CSV：`results/metrics/humaneval-goron-summary.csv`
- 含 RC/RE/Edit/TCP 的完整汇总：
  `results/metrics/humaneval-goron-all-models.{json,csv,md}`
- 论文表格：`../paper/generated_goron_table.tex`

主表报告首个候选的 RC/RE。Nova-1.3B 还报告 20 次采样的无偏 Pass@1
和 Pass@10；Nova-6.7B 报告固定批次 seed 单样本结果；SLaDE 额外报告五
beam oracle。`aggregate.py` 同时生成仅统计
“相对 matched control 确实改变了汇编”的 changed-only 结果，以及按
164 个原始 HumanEval 问题聚类的配对 bootstrap 区间。
