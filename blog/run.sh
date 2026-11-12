#!/bin/bash

# blog/run.sh - data -> baselines -> ours -> results table
#
#   usage: pixi run ./blog/run.sh [dataset ...]          (default: the six dev datasets)
#          SEEDS="0 1 2" JOBS=2 pixi run ./blog/run.sh pets
#
#   Every step skips work that is already on disk, so it is safe to re-run.

set -e
cd "$(dirname "$0")/.."

DATASETS=${@:-pets cars resisc45 flowers food101 fashion_mnist}
SEEDS=${SEEDS:-0 1 2 3 4 5 6 7 8 9}
JOBS=${JOBS:-4}
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3

jobs() { for seed in $SEEDS; do for dataset in $DATASETS; do echo "$dataset $seed"; done; done; }

echo "===== 1. data: download + turn into vectors ====="
for dataset in $DATASETS; do
    python blog/data.py $dataset
done

echo "===== 2. baselines (scikit-learn, graphlearning) ====="
jobs | xargs -P $JOBS -L 1 python blog/baselines.py

echo "===== 3. ours (PPR spreading, cluster-metric graph) ====="
jobs | xargs -P $JOBS -L 1 python blog/ours.py

echo "===== 4. results ====="
python blog/table.py
