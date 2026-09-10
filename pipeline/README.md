# Minimal DeSIR pipeline

The scripts in this directory are deliberately small and composable:

```text
source JSON/JSONL
      │
      ├─ build_dataset.py       clang -O0/-O1/-O2/-O3 + objdump/strings/nm
      │
      ├─ prepare_sft.py         prompt/completion JSONL
      │
      └─ train_sft.py           Transformers Trainer checkpoint

ELF/object + function
      └─ decompile_binary.py    grounding → prompt → C → anchor restoration
```

The grounding schema is versioned as `desir-grounding-v1`. A dataset input row
needs `source` and `function_name`; `id`, `tests`, and `prefix` are optional.
The generated rows contain `assembly`, `grounding`, `c_func`, `c_test`, and
`c_prefix`, so they can be adapted directly to the repository evaluators.

For a model-free smoke test, run:

```bash
clang -O0 -g0 demo.c -o /tmp/desir-demo
python ground_binary.py --binary /tmp/desir-demo --function add_numbers \
  --output /tmp/desir-grounding.json
```

Model inference requires `torch`, `transformers`, and a downloaded checkpoint.
The default public IDs are `xunge/desir-1.3b` and `xunge/desir-6.7b`; a local
path can be supplied with `--model`.
