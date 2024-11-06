#!/bin/bash

# Define the path to the text file containing repository names
CURRENT_DIR=$(pwd)
# file_path="${CURRENT_DIR}/clip_benchmark_clf_datasets.txt"
file_path="${CURRENT_DIR}/test_download_ds.txt"
echo "$file_path"

# Check if the file exists
if [ ! -f "$file_path" ]; then
    echo "File not found: $file_path"
    exit 1
fi

## Pass as argument the path to the directory where the repositories will be cloned, e.g.,
# [BASEPATH_PROJECT]/datasets/wds
cd $1

git lfs install

# Read the file line by line and clone repositories
while IFS= read -r repo_name; do
    # Check if the line is not empty
    if [ -n "$repo_name" ]; then
        echo "Cloning repository: $repo_name"
        git clone "git@hf.co:datasets/clip-benchmark/$repo_name"
        echo "Repository $repo_name cloned."
    fi
done < "$file_path"