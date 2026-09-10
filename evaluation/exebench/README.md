# ExeBench C-only O0--O3 evaluation

All commands below run from `/home/jiang/papers/DecIR/project`. The canonical
scope is the C-only ExeBench v1.01 `test_real` split. A function is retained
only when GCC can produce target-function GAS at O0, O1, O2, and O3 and its
reference source passes all 10 official real I/O pairs. The evaluator reports
compilation rate, individual-I/O accuracy, and the stricter function pass rate
(all 10 tests).

With GCC 13.3.0 on the recorded host, this pre-validation retains 1,941
functions, or 7,764 balanced instances (1,941 per optimization level). The
manifest records all 193 excluded functions and their failure stage.
Wrapper execution uses the ExeBench C++ runtime headers in the locally
reproduced SLaDe artifact at
`../third_party/slade_artifact/slade_module/exebench`.

Build and validate the common dataset:

```bash
python evaluation/exebench/prepare.py \
  --dataset-root /home/jiang/share/dataset/exebench --workers 32
```

Run the four direct checkpoints:

```bash
python evaluation/exebench/run_direct.py \
  --model /home/jiang/model/DecIR-1.3B --name decir-1.3b --gpus 0,1
python evaluation/exebench/run_direct.py \
  --model /home/jiang/model/decir-6.7b --name decir-6.7b --gpus 0,1
python evaluation/exebench/run_direct.py \
  --model /home/jiang/model/llm4decompile-1.3b-v1.5 \
  --name llm4decompile-1.3b-v1.5 --gpus 0,1
python evaluation/exebench/run_direct.py \
  --model /home/jiang/model/llm4decompile-6.7b-v1.5 \
  --name llm4decompile-6.7b-v1.5 --gpus 0,1
```

Run sc²dec, SLaDe, and Nova:

```bash
python evaluation/exebench/run_sccdec.py --gpu 0
python evaluation/exebench/run_slade.py --gpu 0
python evaluation/exebench/run_nova.py --gpus 0,1 --workers-per-gpu 4
python evaluation/exebench/run_nova.py \
  --model /home/jiang/model/nova-6.7b-bcr \
  --architecture-dir ../third_party/nova/nova \
  --predictions results/predictions/nova-6.7b-exebench-c.json \
  --metrics results/metrics/nova-6.7b-exebench-c.json \
  --gpus 0,1 --workers-per-gpu 4
```

SLaDe only releases x86 O0 and O3 checkpoints. Its O1/O2 rows therefore use
the released O3 checkpoint as explicitly labelled zero-shot transfer. Nova
uses its official address-aware object/objdump normalizer and one fixed-seed
sample for the uniform top-1 comparison. Set `--num-return-sequences 20
--generation-batch-size 20` separately when Pass@k candidates are required;
doing so does not change the top-1 table into a Pass@1 estimate unless all 20
candidates are executed.

Regenerate the JSON/CSV/Markdown/LaTeX comparison after all runs:

```bash
python evaluation/summarize_mixed_compiler_c_models.py
python evaluation/summarize_three_benchmark_mixed_compiler.py
```

## Clang-input DecIR rerun

The compiler-sensitivity rerun changes only the compiler that produces model
input assembly. Clang 18.1.3 emits target-function GAS at O0--O3, while the
generated C candidates and official ExeBench wrappers retain the common
GCC/g++ 13.3.0 execution oracle. This isolates input-codegen effects without
changing the accepted wrapper scope at the same time.

Build the balanced Clang-input scope:

```bash
python evaluation/exebench/prepare.py \
  --dataset-root /home/jiang/share/dataset/exebench \
  --output decompile-eval/exebench-test-real-c-clang-o0-o3.json \
  --metadata decompile-eval/exebench-test-real-c-clang-metadata.json \
  --reference-validation \
    results/metrics/exebench-clang-input-gcc-oracle-reference-validation.json \
  --cc clang --cxx clang++ \
  --validation-cc gcc --validation-cxx g++ --workers 32
```

This retains 1,933 validated C functions, or 7,732 rows (1,933 per
optimization level). Run both DecIR checkpoints:

```bash
python evaluation/exebench/run_direct.py \
  --model /home/jiang/model/DecIR-1.3B --name decir-1.3b-clang-input \
  --dataset decompile-eval/exebench-test-real-c-clang-o0-o3.json \
  --metadata decompile-eval/exebench-test-real-c-clang-metadata.json \
  --predictions results/predictions/decir-1.3b-exebench-clang-c.json \
  --metrics results/metrics/decir-1.3b-exebench-clang-c.json \
  --gpus 0,1 --cc gcc --cxx g++

python evaluation/exebench/run_direct.py \
  --model /home/jiang/model/decir-6.7b --name decir-6.7b-clang-input \
  --dataset decompile-eval/exebench-test-real-c-clang-o0-o3.json \
  --metadata decompile-eval/exebench-test-real-c-clang-metadata.json \
  --predictions results/predictions/decir-6.7b-exebench-clang-c.json \
  --metrics results/metrics/decir-6.7b-exebench-clang-c.json \
  --gpus 0,1 --cc gcc --cxx g++
```

After the MBPP Clang-input runs also finish, generate the combined DecIR
JSON/CSV/Markdown report and LaTeX rows:

```bash
python evaluation/summarize_clang_decir.py
python evaluation/summarize_mixed_compiler_c_models.py
```

The second command combines the Clang DecIR rows with the existing GCC
LLM4Decompile, sc²dec, SLaDe, and Nova rows, and additionally computes
ExeBench normalized edit similarity.

## DecIR-6.7B-STR Clang binary evaluation

Create binary-grounded STR inputs for the same 1,933-function C-only scope.
The three legacy `SASAtomic*` sources use a GCC-style pointer constraint that
Clang expands to invalid double parentheses; the preparer records and applies
an address-equivalent register-constraint compatibility fix to those 12 rows.

```bash
python evaluation/prepare_string_dataset.py \
  --benchmark exebench \
  --dataset decompile-eval/exebench-test-real-c-clang-o0-o3.json \
  --output decompile-eval/exebench-test-real-c-clang-string-aware-o0-o3.json \
  --workers 32

# Run these two commands concurrently on GPU 0 and GPU 1.
CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
python evaluation/run_inference_vllm_string.py \
  --model /home/jiang/model/decir-6.7b-str \
  --dataset decompile-eval/exebench-test-real-c-clang-string-aware-o0-o3.json \
  --output results/predictions/decir-6.7b-str-exebench-clang-c.shard-0-of-2.json \
  --shard-index 0 --num-shards 2 --shard-group-size 4

CUDA_VISIBLE_DEVICES=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
python evaluation/run_inference_vllm_string.py \
  --model /home/jiang/model/decir-6.7b-str \
  --dataset decompile-eval/exebench-test-real-c-clang-string-aware-o0-o3.json \
  --output results/predictions/decir-6.7b-str-exebench-clang-c.shard-1-of-2.json \
  --shard-index 1 --num-shards 2 --shard-group-size 4

python evaluation/merge_prediction_shards.py \
  --dataset decompile-eval/exebench-test-real-c-clang-string-aware-o0-o3.json \
  --inputs results/predictions/decir-6.7b-str-exebench-clang-c.shard-{0,1}-of-2.json \
  --output results/predictions/decir-6.7b-str-exebench-clang-c.json

python evaluation/exebench/evaluate.py \
  --dataset decompile-eval/exebench-test-real-c-clang-string-aware-o0-o3.json \
  --metadata decompile-eval/exebench-test-real-c-clang-metadata.json \
  --predictions results/predictions/decir-6.7b-str-exebench-clang-c.json \
  --output results/metrics/decir-6.7b-str-exebench-clang-c.json \
  --workers 64 --cc gcc --cxx g++ --objcopy objcopy \
  --bootstrap-samples 10000 --seed 42
```

The attachment's default 8,192-token context and 1,024-token output reserve
are preserved. Inputs exceeding the remaining 7,168-token prompt budget are
recorded as context-overflow failures rather than silently truncated.

The completed C-only averages are RC 92.62%, strict 10/10 RE 74.16%, Edit
48.76%, and TCP 76.05%. The per-optimization table is in
`results/metrics/decir-6.7b-str-clang-mbpp-exebench-summary.md`.
