#!/bin/bash

# Exit on error
set -e

CURRENT_DIR=$(pwd)
test_model_config="${CURRENT_DIR}/test_models_config.json"
test_datasets="${CURRENT_DIR}/test_webdatasets_w_in10k.txt"

cd ..

python  distance_matrix_computation.py \
       --models_config="$test_model_config" \
       --datasets="$test_datasets"
