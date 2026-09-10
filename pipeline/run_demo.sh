#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/demo-output"
mkdir -p "${OUT}"

clang -O0 -g0 "${ROOT}/pipeline/demo.c" -o "${OUT}/demo"
python "${ROOT}/pipeline/ground_binary.py" \
  --binary "${OUT}/demo" --function add_numbers \
  --output "${OUT}/grounding.json"

python "${ROOT}/pipeline/decompile_binary.py" \
  --binary "${OUT}/demo" --function add_numbers \
  --model "${DESIR_MODEL:-xunge/desir-1.3b}" \
  --output "${OUT}/decompilation.json"

printf 'Saved result to %s\n' "${OUT}/decompilation.json"
