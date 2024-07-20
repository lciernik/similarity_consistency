import os
import argparse
import json
import sys
from helper import parse_datasets
import pandas as pd
from clip_benchmark.analysis.utils import retrieve_performance 

parser = argparse.ArgumentParser()
parser.add_argument('--model_config_dict', type=str, default="./models_config.json")
parser.add_argument('--dataset', type=str, default="./webdatasets.txt")    
parser.add_argument('--output_root', type=str, default="./test_results/max_performance_per_tuned_model")
args = parser.parse_args()

datasets = parse_datasets(args.dataset)
datasets = [ds.replace('/', '_')for ds in datasets]

with open(args.model_config_dict, 'r') as f:
            models = json.load(f)

for dataset in datasets:
    max_performance = {}
    for model_key in models.keys():
        try:
            performance = retrieve_performance(model_id=model_key, dataset_id=dataset)
            max_performance[model_key] = performance
        except (pd.errors.DatabaseError, FileNotFoundError) as e:
            continue

    max_performance = dict(sorted(max_performance.items(), key= lambda x: x[1], reverse=True))
    
    out_fn = os.path.join(args.output_root, f'max_performance_per_model_{dataset}.json')
    with open(out_fn, 'w') as f:
        json.dump(max_performance, f, indent=2)    
        print(f"Storing file: {out_fn=}")