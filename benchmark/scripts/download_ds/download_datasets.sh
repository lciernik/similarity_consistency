#!/bin/bash

# Define the path to the text file containing repository names
# file_path="/home/lciernik/projects/divers-priors/diverse_priors/benchmark/scripts/download_ds/vtab_ds_origname.txt"
file_path="/home/lciernik/projects/divers-priors/diverse_priors/benchmark/scripts/download_ds/other_clf_ds_origname.txt"

# Check if the file exists
if [ ! -f "$file_path" ]; then
    echo "File not found: $file_path"
    exit 1
fi

cd /home/space/diverse_priors/datasets/wds

# Read the file line by line and clone repositories
while IFS= read -r repo_name; do
    # Check if the line is not empty
    if [ -n "$repo_name" ]; then
        echo "Cloning repository: $repo_name"
        git clone "git@hf.co:datasets/clip-benchmark/$repo_name"
        echo "Repository $repo_name cloned."
    fi
done < "$file_path"