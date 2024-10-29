import pandas as pd
import json 

def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

PCT_CUTOFF = 0.8 

fn_performances = './test_results/max_performance_per_tuned_model/max_performance_per_model_wds_imagenet1k.json'
fn_models_config = './models_config.json'

performances = load_json(fn_performances)
performances = pd.DataFrame(performances.items(), columns=['model_id', 'test_1acc'])
performances.sort_values('test_1acc', ascending=False, inplace=True)

threshold = performances['test_1acc'].iloc[0] * PCT_CUTOFF
performances = performances[performances['test_1acc'] >= threshold].copy()

models_config = load_json(fn_models_config)

filtered_models_config = {k:v for k, v in models_config.items() if k in performances['model_id'].tolist()}

with open(f"filtered_models_config.json", "w") as f:
    json.dump(filtered_models_config, f, indent=4)

