# DeSIR

DeSIR is an assembly-to-C decompilation system that uses a language model with compiler-generated intermediate representations and STR/CONST/DATA anchors as auxiliary supervision. This source-only release contains inference, data preparation, training, evaluation, and examples, but no model weights, generated predictions, private datasets, or third-party repositories.

## Layout

```text
dataset/                         Dataset and LLVM-IR/assembly builders
pipeline/                        Minimal source-to-binary-to-C reference flow
evaluation/desir/                DeSIR anchor-aware vLLM inference
evaluation/humaneval/            HumanEval-Decompile protocol helpers
evaluation/mbpp/                 MBPP-Decompile C protocol helpers
evaluation/exebench/             ExeBench C protocol helpers
evaluation/common.py             Shared compile/run utilities
evaluation/evaluate_*.py         RC, RE, TCP, and benchmark evaluators
train/                           Optional training and dataset packing code
examples/                        Binary-to-source demonstration programs
tests/                            Lightweight regression tests
```

## Installation

Use a Python environment with the versions in `requirements.txt` for inference and evaluation. Training additionally needs `requirements-train.txt`.

```bash
cd /home/jiang/projects/DeSIR
python -m pip install -r requirements.txt
```

The vLLM evaluator expects CUDA. The compiler, `objdump`, and `clang-format` are external system dependencies.

## Inference

The generic evaluator accepts a prepared JSON test set. For anchor-aware DeSIR inputs, use the provided preparation script or pass an already prepared dataset with `--string_prompt_mode existing`:

```bash
cd /home/jiang/projects/DeSIR
python evaluation/desir/prepare_goron_desir_dataset.py \
  --input /path/to/humaneval-decompile-goron.json \
  --output /path/to/humaneval-decompile-goron-desir.json \
  --compiler clang

CUDA_VISIBLE_DEVICES=0,1 python evaluation/desir/run_evaluation_desir_vllm.py \
  --model_path /path/to/desir-checkpoint \
  --testset_path /path/to/humaneval-decompile-goron-desir.json \
  --output_path results/predictions/desir.json \
  --output_result_path results/metrics/desir.json \
  --compiler clang --gpus 2 \
  --string_prompt_mode existing
```

For a newly compiled STR/CONST/DATA dataset, use `--string_prompt_mode rebuild`, `--anchor_preprocessor_path dataset/1_get_exebench_asm_llvmir.py`, and the appropriate compiler override. Checkpoint paths are command-line arguments; weights are never redistributed here.

## Metrics

`evaluation/evaluate_predictions.py` computes recompilability (RC) and re-executability (RE). `evaluation/evaluate_tcp.py` computes macro-averaged test-case pass rate (TCP), and `evaluation/evaluate_decompile_bench.py` adds the upstream-compatible execution protocol and edit similarity.

```bash
python evaluation/evaluate_predictions.py \
  --predictions results/predictions/desir.json \
  --output results/metrics/desir-rc-re.json \
  --compiler clang

python evaluation/evaluate_tcp.py \
  --predictions results/predictions/desir.json \
  --output results/metrics/desir-tcp.json \
  --compiler clang
```

The benchmark-specific READMEs under `evaluation/humaneval/`, `evaluation/mbpp/`, and `evaluation/exebench/` document input schemas, compiler protocols, optimization levels, and aggregation rules.

## Training

The optional training pipeline is based on ColossalAI and LLaMA-compatible Transformers models. Prepare packed Arrow data with `train/prepare_pretrain_dataset.py`, set `PRETRAINED_MODEL` and `TRAIN_DATA`, then run:

```bash
cd /home/jiang/projects/DeSIR/train
PRETRAINED_MODEL=/path/to/base-model \
TRAIN_DATA=/path/to/arrow-dataset \
NPROC=2 ./run_train.sh
```

Review the shell scripts before launching a long run and set output paths, seed, sequence length, and GPU count for the local machine.

## Data and model provenance

The source tree intentionally excludes checkpoint files and benchmark data. Obtain authorized DeSIR checkpoints and HumanEval-Decompile, MBPP-Decompile, and ExeBench inputs separately, then pass their paths to the commands above. Do not commit credentials, downloaded weights, generated predictions, or binaries.

## Minimal end-to-end binary demo

The `pipeline/` directory is a self-contained reference implementation of the
complete flow:

1. `ground_binary.py` runs `objdump`, `strings`, and `nm` to extract the target
   function and binary-derived STR/DATA grounding records.
2. `decompile_binary.py` renders the assembly plus grounding into the DeSIR
   prompt, loads a checkpoint with Transformers, generates C, and restores
   anchor tokens.
3. The JSON output stores the prompt, raw generation, restored source, and
   grounding for reproducibility.

The public checkpoints can be selected directly with their Hugging Face IDs:

```text
xunge/desir-1.3b
xunge/desir-6.7b
```

Run the complete example after installing the dependencies and logging into
Hugging Face if the model is gated:

```bash
cd /home/jiang/projects/DeSIR
DESIR_MODEL=xunge/desir-1.3b pipeline/run_demo.sh
# or use xunge/desir-6.7b on a sufficiently large GPU
```

To inspect only static grounding without loading a model:

```bash
python pipeline/ground_binary.py --binary ./demo-output/demo \
  --function add_numbers --output ./demo-output/grounding.json
```

For training, first construct optimization-specific examples from a source
JSON/JSONL file containing `id`, `source`, and `function_name`:

```bash
python pipeline/build_dataset.py --input sources.json \
  --output data/desir-grounded.json --compiler clang
python pipeline/prepare_sft.py --input data/desir-grounded.json \
  --output data/desir-sft.jsonl
python pipeline/train_sft.py --model deepseek-ai/deepseek-coder-1.3b-base \
  --train data/desir-sft.jsonl --output runs/desir-1.3b
```

`build_dataset.py` creates O0/O1/O2/O3 rows and preserves optional `tests` and
`prefix` fields. `evaluate_predictions.py` and `evaluate_tcp.py` can then be
used on generated rows to obtain RC/RE and TCP. The minimal trainer uses the
standard Transformers `Trainer`; for 6.7B training, use distributed launch,
activation checkpointing, or a PEFT/LoRA configuration appropriate for the
available GPUs.

## License and attribution

This release contains original DeSIR glue/evaluation code and small adapted utilities. Check the upstream licenses of PyTorch, Transformers, vLLM, ColossalAI, and each benchmark before redistribution. Add the project license and required third-party notices before publishing a public repository.
