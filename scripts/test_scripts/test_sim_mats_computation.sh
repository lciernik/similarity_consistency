#!/bin/bash

# Exit on error
set -e

test_model_config="test_scripts/test_models_config.json"
test_datasets="test_scripts/test_webdatasetst_w_in10k.txt"

cd ..

python  distance_matrix_computation.py \
       --models_config="$test_model_config" \
       --datasets="$test_datasets"
