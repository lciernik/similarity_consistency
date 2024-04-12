import os
import sys
import json
from datetime import datetime
import random
from itertools import product

sys.path.append('..')
from slurm import run_job
from helper import load_models, get_hyperparams

MODELS_CONFIG = "../models_config.json"
DATASETS = "./webdatasets_test.txt"
DATASETS_ROOT = "/home/space/diverse_priors/datasets/wds/wds_{dataset_cleaned}"

# Create new test experiment folder
BASE_PATH_EXP = "./test_results"
current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M")
BASE_PATH_EXP = os.path.join(BASE_PATH_EXP, current_datetime)
os.makedirs(BASE_PATH_EXP, exist_ok=True)

FEATURES_ROOT = os.path.join(BASE_PATH_EXP, 'features')
OUTPUT_ROOT = os.path.join(BASE_PATH_EXP, 'results', 'single_models', '{fewshot_k}', '{dataset}', '{model}',
                           'fewshot_lr_{fewshot_lr}', 'fewshot_epochs_{fewshot_epochs}', 'seed_{seed}')

if __name__ == "__main__":
    # Select random model
    models, n_models = load_models(MODELS_CONFIG)
    random_model = random.choice(list(models.keys()))

    # Select random hyperparameters
    hyper_params, _ = get_hyperparams(num_seeds=10)
    hyper_params = {k: [random.choice(v)] for k, v in hyper_params.items()}
    num_jobs = len(list(product(*hyper_params.values())))

    print(f"Testing single model {random_model} with hyper_params: {hyper_params}")

    # Evaluate each model on all datasets and all hyperparameter configurations.
    job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
    clip_benchmark eval --dataset {DATASETS} \
                        --dataset_root {DATASETS_ROOT} \
                        --feature_root {FEATURES_ROOT} \
                        --output {OUTPUT_ROOT} \
                        --task=linear_probe \
                        --model {models[random_model]['model_name']} \
                        --model_source {models[random_model]['source']} \
                        --model_parameters '{json.dumps(models[random_model]['model_parameters'])}' \
                        --module_name {models[random_model]['module_name']} \
                        --batch_size=64 \
                        --fewshot_k {' '.join(hyper_params['fewshot_ks'])} \
                        --fewshot_lr {' '.join(hyper_params['fewshot_lrs'])} \
                        --fewshot_epochs {' '.join(hyper_params['fewshot_epochs'])} \
                        --train_split train \
                        --test_split test \
                        --seed {' '.join(hyper_params['seeds'])} 
    """

    run_job(
        job_name=f"test_single",
        job_cmd=job_cmd,
        partition='gpu-2h',
        log_dir='./logs',
        num_jobs_in_array=num_jobs
    )
