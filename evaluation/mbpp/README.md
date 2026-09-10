# DecIR-6.7B on MBPP-Decompile

从 `/home/jiang/papers/DecIR/project` 运行：

```bash
python evaluation/mbpp/decir_6_7b.py --gpus 0,1
```

入口默认加载实际存在的 `/home/jiang/model/decir-6.7b`，以 MBPP 问题为
单位将 C/C++ 和 O0--O3 均衡交错分配到两块 GPU。合并后使用参考验证文件
排除所有方法共同排除的 32 条无效上游 C++ 样本，最终在 7,760 条样本上
报告 RC、RE、edit similarity、按优化级别和语言的结果，以及按 974 个
MBPP 问题聚类的 bootstrap 区间。

当前规范评分结果为 RC 44.32%、RE 21.56%、edit similarity 31.20%。
对固定预测执行三次完整评分得到 21.56%--21.60% RE；差异仅来自 3 个含
数组越界读取的生成候选。规范结果保留较保守的 21.56% 单次执行值。

已有完整预测时会跳过推理并重新评分。强制重新推理：

```bash
python evaluation/mbpp/decir_6_7b.py --gpus 0,1 --force-inference
```

默认输出：

- `results/predictions/decir-6.7b-mbpp-decompile.json`
- `results/metrics/decir-6.7b-mbpp-decompile.json`
- `results/metrics/decir-6.7b-vs-1.3b-mbpp-paired.json`
- `results/metrics/decir-6.7b-vs-llm4decompile-v1.5-mbpp-paired.json`
- `results/metrics/decir-6.7b-vs-llm4decompile-v1.6-mbpp-paired.json`
- `../paper/generated_mbpp_6_7b.tex`

## C-only O0--O3 comparison

The canonical C slice contains 3,896 instances: 974 MBPP problems at each of
O0, O1, O2, and O3. Generate and score the previously missing
LLM4Decompile-6.7B-v1.5 row with:

```bash
python evaluation/mbpp/direct_c.py --gpus 0,1
```

SLaDe only releases x86 O0 and O3 checkpoints. The following command preserves
those native mappings and explicitly evaluates O1/O2 as zero-shot transfer
through the O3 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run_inference_slade.py \
  --dataset decompile-eval/mbpp-decompile.json \
  --output results/predictions/slade-released-x86-allopts-mbpp-c.json \
  --o0-model ../third_party/slade_artifact/output/export-new_train-2023-04-09-1854-b799-06dc-checkpoint_best \
  --o3-model ../third_party/slade_artifact/output/export-new_train-2023-04-28-2313-1dff-748d-checkpoint_best \
  --use-o3-for-o1-o2 --assembly-source gcc-source --device cuda:0 \
  --batch-size 16 --num-beams 5 --num-return-sequences 5

python evaluation/evaluate_decompile_bench.py \
  --predictions results/predictions/slade-released-x86-allopts-mbpp-c.json \
  --output results/metrics/slade-released-x86-allopts-mbpp-c.json \
  --benchmark-metadata decompile-eval/mbpp-decompile.json \
  --reference-validation results/metrics/mbpp-decompile-reference-validation.json \
  --workers 64 --bootstrap-samples 10000 --seed 42 \
  '--crypto-library=-l:libcrypto.so.3'

python evaluation/mbpp/run_nova.py \
  --model /home/jiang/model/nova-6.7b-bcr \
  --architecture-dir ../third_party/nova/nova \
  --gpus 0,1 --workers-per-gpu 4
```

After all metric artifacts exist, regenerate the JSON, CSV, Markdown, and
LaTeX comparison tables:

```bash
python evaluation/mbpp/summarize_c_models.py
```

The completed comparison is in
`results/metrics/mbpp-c-model-comparison.{json,csv,md}`. Each optimization
level contains 974 C instances. Average RE is 23.92%/32.65% for
DecIR-1.3B/6.7B and 32.26%/47.69% for
LLM4Decompile-1.3B/6.7B-v1.5.

## Clang-input C-only rerun

Rebuild the C-only inputs with Clang 18.1.3 at each optimization level. The
script links the supplied reference harness into an executable before
extracting the target function through GNU objdump. Linking is required so
that external and local call relocations are resolved; disassembling a
relocatable object with plain `objdump -d` incorrectly renders unresolved
library calls as targets inside `func0`.

```bash
python evaluation/mbpp/prepare_clang.py \
  --source decompile-eval/mbpp-decompile-c.json \
  --output decompile-eval/mbpp-decompile-clang-linked-c.json \
  --clang clang --objdump objdump --workers 64
```

One upstream C problem uses a GCC-only nested-function extension, so the
balanced Clang scope contains 973 problems and 3,892 rows (973 per O-level).
The generated candidates retain the common GCC/g++ execution oracle.

```bash
python evaluation/mbpp/run_clang.py \
  --model /home/jiang/model/DecIR-1.3B --name decir-1.3b \
  --gpus 0,1 --score-cc gcc --score-cxx g++

python evaluation/mbpp/run_clang.py \
  --model /home/jiang/model/decir-6.7b --name decir-6.7b \
  --gpus 0,1 --score-cc gcc --score-cxx g++

python evaluation/summarize_clang_decir.py
python evaluation/summarize_mixed_compiler_c_models.py
```

The final combined report is
`results/metrics/decir-clang-c-comparison.{json,csv,md}`; the paper rows are
written to `../paper/generated_decir_clang_c.tex`.
The requested eight-model mixed-input-compiler report is written to
`results/metrics/mixed-compiler-c-model-comparison.{json,csv,md}`.
The complete three-benchmark, eight-model report is written by
`python evaluation/summarize_three_benchmark_mixed_compiler.py` to
`results/metrics/three-benchmark-mixed-compiler-comparison.{json,csv,md}`.
The linked-input predictions and metrics are:

- `results/predictions/decir-1.3b-mbpp-clang-linked-c.json`
- `results/predictions/decir-6.7b-mbpp-clang-linked-c.json`
- `results/metrics/decir-1.3b-mbpp-clang-linked-c.json`
- `results/metrics/decir-6.7b-mbpp-clang-linked-c.json`

The older `*-mbpp-clang-c.json` artifacts are retained only to diagnose the
invalid unlinked-object protocol and must not be reported as Clang results.

## DecIR-6.7B-STR Clang binary evaluation

The STR checkpoint uses the exact prompt and deterministic string-restoration
protocol from `/home/jiang/projects/DecIR/evaluation_string/decompile_binary.py`.
Build string-aware inputs from the linked 973-function C-only scope, run one
interleaved shard on each GPU, merge, and score with:

```bash
python evaluation/prepare_string_dataset.py \
  --benchmark mbpp \
  --dataset decompile-eval/mbpp-decompile-clang-linked-c.json \
  --output decompile-eval/mbpp-decompile-clang-string-aware-c.json \
  --workers 24

CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
python evaluation/run_inference_vllm_string.py \
  --model /home/jiang/model/decir-6.7b-str \
  --dataset decompile-eval/mbpp-decompile-clang-string-aware-c.json \
  --output results/predictions/decir-6.7b-str-mbpp-clang-c.shard-0-of-2.json \
  --shard-index 0 --num-shards 2 --shard-group-size 4

CUDA_VISIBLE_DEVICES=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
python evaluation/run_inference_vllm_string.py \
  --model /home/jiang/model/decir-6.7b-str \
  --dataset decompile-eval/mbpp-decompile-clang-string-aware-c.json \
  --output results/predictions/decir-6.7b-str-mbpp-clang-c.shard-1-of-2.json \
  --shard-index 1 --num-shards 2 --shard-group-size 4

python evaluation/merge_prediction_shards.py \
  --dataset decompile-eval/mbpp-decompile-clang-string-aware-c.json \
  --inputs results/predictions/decir-6.7b-str-mbpp-clang-c.shard-{0,1}-of-2.json \
  --output results/predictions/decir-6.7b-str-mbpp-clang-c.json

python evaluation/evaluate_decompile_bench.py \
  --predictions results/predictions/decir-6.7b-str-mbpp-clang-c.json \
  --prediction-field output \
  --output results/metrics/decir-6.7b-str-mbpp-clang-c.json \
  --benchmark-metadata decompile-eval/mbpp-decompile-clang-string-aware-c.json \
  --reference-validation results/metrics/mbpp-decompile-reference-validation.json \
  --c-compiler gcc --cpp-compiler g++ --workers 64 \
  --bootstrap-samples 10000 --seed 42 \
  '--crypto-library=-l:libcrypto.so.3'

python evaluation/evaluate_tcp.py \
  --benchmark mbpp --model-name DecIR-6.7B-STR \
  --input-compiler 'Clang 18.1.3 string-aware binary' \
  --predictions results/predictions/decir-6.7b-str-mbpp-clang-c.json \
  --prediction-field output \
  --output results/metrics/tcp/decir-6.7b-str-mbpp-clang-c.json \
  --compiler gcc --workers 32
```

The completed C-only averages are RC 87.08%, RE 63.59%, Edit 47.22%, and
TCP 66.44%. The per-optimization table is in
`results/metrics/decir-6.7b-str-clang-mbpp-exebench-summary.md`.
