# Decompile named functions from a binary

These examples accept an x86-64 ELF binary and either:

- `--functions name1 name2 ...`; or
- `--functions-file functions.txt`, with one ELF symbol per line (a JSON
  string array is also accepted).

Each script prints the generated functions and writes an auditable JSON file.
The JSON retains the model-specific assembly representation, all returned
candidates, errors, and generation settings. Commands below run from the
`project` directory.

Build the included smoke-test binary if a quick input is needed:

```bash
gcc -O3 -fno-inline examples/sample_binary.c -o /tmp/decir-sample
nm -n /tmp/decir-sample | grep -E ' (checksum|clamp|main)$'
```

Run all five methods on the included `checksum` and `clamp` functions:

```bash
gcc -O3 -fno-inline -fno-omit-frame-pointer \
  examples/sample_binary.c -o results/binary_demo/sample_binary
bash examples/run_binary_demo.sh
```

List available symbols before inference:

```bash
nm -n /path/to/program | less
```

For C++ use the raw/mangled symbol consumed by `objdump`, not the name printed
only by `nm -C`. A fully stripped binary has no function-name boundary oracle;
these examples deliberately fail instead of guessing boundaries.

## DecIR

```bash
CUDA_VISIBLE_DEVICES=0 python examples/decompile_decir.py \
  --binary /path/to/program \
  --functions main helper checksum \
  --output results/binary/decir.json
```

The default model is `/home/jiang/model/DecIR-1.3B`.

## LLM4Decompile

```bash
CUDA_VISIBLE_DEVICES=0 python examples/decompile_llm4decompile.py \
  --binary /path/to/program \
  --functions main helper checksum \
  --output results/binary/llm4decompile.json
```

The default model is
`/home/jiang/model/llm4decompile-1.3b-v1.5`. Pass a different compatible
checkpoint with `--model`.

## sc²dec

```bash
CUDA_VISIBLE_DEVICES=0 python examples/decompile_sccdec.py \
  --binary /path/to/program \
  --functions main helper checksum \
  --optimization O3 \
  --output results/binary/sccdec.json
```

This runs the released FAE LoRA over
`/home/jiang/model/llm4decompile-6.7b-v1.5`, with the author's one-shot
example. It then recompiles each initial `func0` prediction and, when
compilation succeeds, runs the self-constructed-context second pass. The JSON
contains both `initial_source` and the final `source`, plus
`self_context_used`.

If another process already occupies most of the GPU, disable CUDA graphs and
set a smaller explicit memory fraction, provided enough memory remains for the
6.7B weights and KV cache:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/decompile_sccdec.py \
  --binary /path/to/program --functions helper --optimization O3 \
  --enforce-eager --gpu-memory-utilization 0.20 \
  --max-model-len 4096 --output results/binary/sccdec.json
```

## SLaDe

```bash
CUDA_VISIBLE_DEVICES=0 python examples/decompile_slade.py \
  --binary /path/to/program \
  --functions main helper checksum \
  --optimization O3 \
  --output results/binary/slade.json
```

Only `O0` and `O3` are valid because those are the released x86 checkpoints.
SLaDe was trained on compiler-produced GAS, whose source directives and local
labels do not survive in a binary. The example reconstructs local jump labels
from `objdump` and wraps the AT&T instructions in SLaDe's GAS envelope. This
adapter is useful for trying a real binary, but it is not identical to the
authors' compiler-GAS evaluation protocol; that limitation is recorded in the
output.

Use `--num-return-sequences 5` to retain all five released beam-search
hypotheses.

## Nova

Nova requires the pinned Transformers 4.40 environment:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-nova/bin/python examples/decompile_nova.py \
  --binary /path/to/program \
  --functions main helper checksum \
  --optimization O3 \
  --output results/binary/nova.json
```

The example calls the authors' `data/normalize.py`, custom tokenizer,
two-dimensional attention mask, and released `modeling_nova.py`; it does not
treat Nova as a stock causal LM. Nova keeps at most the first 257 normalized
assembly lines. Its default is one sampled candidate using the authors'
temperature/top-p values. Use `--temperature 0` for one deterministic greedy
candidate, or `--num-return-sequences 20` for the paper-style sampled set.

## Python API

The scripts also expose a `decompile` function:

```python
from examples.decompile_decir import decompile

binary, records = decompile(
    "/path/to/program",
    ["main", "helper"],
    model_path="/home/jiang/model/DecIR-1.3B",
)
sources = {item["function_name"]: item["source"] for item in records}
```

The other modules expose the same first two arguments:

```python
from examples.decompile_llm4decompile import decompile
from examples.decompile_sccdec import decompile
from examples.decompile_slade import decompile
from examples.decompile_nova import decompile
```

For Nova, SLaDe, and sc²dec, pass the binary's real optimization level because
it is part of their input or self-context protocol. Optimization level cannot
be recovered reliably from arbitrary machine code.

## Scope and safety

- Input must be an ELF64 x86-64 executable, shared object, or object file.
- Function names must still be present. Function discovery and recovery from
  a stripped binary are separate reverse-engineering tasks.
- Output signatures and external declarations are model predictions and may
  be wrong; generated source is not automatically safe or executable.
- Do not compile or run generated code outside an isolated environment.
