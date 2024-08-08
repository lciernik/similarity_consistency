import os
import argparse
import json
import sys
from helper import parse_datasets
import pandas as pd
from clip_benchmark.analysis.utils import retrieve_performance 
from clip_benchmark.tasks.linear_probe import Regularization

parser = argparse.ArgumentParser()
parser.add_argument('--model_config_dict', type=str, default="./models_config.json")
parser.add_argument('--dataset', type=str, default="./webdatasets.txt")    
parser.add_argument('--regularization', default=["weight_decay", "L1", "L2"], type=str, nargs='+', help="Type of regularization applied during training.", choices=[reg.value for reg in Regularization])
parser.add_argument('--output_root', type=str, default="/home/space/diverse_priors/results/aggregated/max_performance_per_model_n_ds")
parser.add_argument('--force_computation', action='store_true')
args = parser.parse_args()

datasets = parse_datasets(args.dataset)
datasets = [ds.replace('/', '_')for ds in datasets]

with open(args.model_config_dict, 'r') as f:
            models = json.load(f)

for dataset in datasets:
    for reg_method in args.regularization:
        print(f"Gathering results for {dataset=} and {reg_method=}")
        
        curr_storing_path = os.path.join(args.output_root, reg_method)
        if not os.path.exists(curr_storing_path):
            os.makedirs(curr_storing_path)

        out_fn = os.path.join(curr_storing_path , f'{dataset}.json')
        if not args.force_computation and os.path.isfile(out_fn):
            print(f"{out_fn=} already exists! Continuing ...")
            continue

        max_performance = {}
        for model_key in models.keys():
            try:
                performance = retrieve_performance(model_id=model_key, dataset_id=dataset, regularization=reg_method)
                max_performance[model_key] = performance
            except (pd.errors.DatabaseError, FileNotFoundError) as e:
                print(e)
                continue
        
        if not max_performance:
            print(f"max_performance is empty! Not storing and continuing")
            continue
        
        max_performance = dict(sorted(max_performance.items(), key= lambda x: x[1], reverse=True))
        
        with open(out_fn, 'w') as f:
            json.dump(max_performance, f, indent=2)    
            print(f"Storing file: {out_fn=}")