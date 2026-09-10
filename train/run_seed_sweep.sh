#!/usr/bin/env bash
set -euo pipefail

# Launch sequential, reproducible training runs.  Override SEEDS to change the
# preregistered set, e.g. SEEDS="1 2 3 4 5".
SEEDS="${SEEDS:-41 42 43}"

for seed in ${SEEDS}; do
    RUN_NAME="${RUN_PREFIX:-decir-safe}-seed${seed}" \
    SEED="${seed}" \
    MASTER_PORT="$((30000 + seed % 1000))" \
    bash run_train.sh
done
