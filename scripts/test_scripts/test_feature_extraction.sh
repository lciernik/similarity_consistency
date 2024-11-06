#!/bin/bash

# Exit on error
set -e

test_model_config="test_scripts/test_models_config.json"
test_datasets="test_scripts/test_webdatasets.txt"

cd ..

python feature_extraction.py \
       --models_config="$test_model_config" \
       --datasets="$test_datasets"
