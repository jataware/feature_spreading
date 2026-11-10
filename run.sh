#!/bin/bash

# run.sh
#   usage: pixi run ./run.sh [dataset ...]   (default: all datasets)

set -e

DATASETS=${@:-pets cars resisc45 flowers fashion_mnist food101}

for dataset in $DATASETS; do
    echo "===== $dataset ====="
    fs download $dataset
    fs run $dataset
    fs agop $dataset
done
