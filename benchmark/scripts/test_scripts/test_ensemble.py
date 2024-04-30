import os
import sys
import json
from datetime import datetime
import random
from itertools import product

sys.path.append('..')
from slurm import run_job
from helper import load_models, get_hyperparams, prepare_for_combined_usage

MODELS_CONFIG = "./test_models_config.json"
DATASETS = "imagenet-subset-10k"  # "./webdatasets_test.txt"
DATASETS_ROOT = "/home/space/diverse_priors/datasets/wds/wds_{dataset_cleaned}"
FEATURES_ROOT = "/home/space/diverse_priors/features"

# Create new test experiment folder
BASE_PATH_EXP = "./test_results"
current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M")
BASE_PATH_EXP = os.path.join(BASE_PATH_EXP, current_datetime)
os.makedirs(BASE_PATH_EXP, exist_ok=True)

OUTPUT_ROOT = os.path.join(BASE_PATH_EXP, 'results', '{task}', '{fewshot_k}', '{dataset}', '{model}',
                           'fewshot_lr_{fewshot_lr}', 'fewshot_epochs_{fewshot_epochs}', 'seed_{seed}')

if __name__ == "__main__":
    # Two random models to evaluate in combination
    models, n_models = load_models(MODELS_CONFIG)
    models2combine = random.sample(list(models.keys()), 2)
    models = {k: v for k, v in models.items() if k in models2combine}
    # Prepare job command input
    model_names, sources, model_parameters, module_names = prepare_for_combined_usage(models)

    # Select random hyperparameters
    hyper_params, _ = get_hyperparams(num_seeds=10)
    hyper_params = {k: [random.choice(v)] for k, v in hyper_params.items()}
    num_jobs = len(list(product(*hyper_params.values())))

    print(f"Testing ensembling models {models2combine} with hyper_params {hyper_params}")

    job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
            export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
            clip_benchmark eval --dataset {DATASETS} \
                                --dataset_root {DATASETS_ROOT} \
                                --feature_root {FEATURES_ROOT} \
                                --output {OUTPUT_ROOT} \
                                --task=ensembling \
                                --model {' '.join(model_names)} \
                                --model_source {' '.join(sources)} \
                                --model_parameters {' '.join([f"'{json.dumps(x)}'" for x in model_parameters])} \
                                --module_name {' '.join(module_names)} \
                                --batch_size=64 \
                                --fewshot_k {' '.join(hyper_params['fewshot_ks'])} \
                                --fewshot_lr {' '.join(hyper_params['fewshot_lrs'])} \
                                --fewshot_epochs {' '.join(hyper_params['fewshot_epochs'])} \
                                --train_split train \
                                --test_split test \
                                --seed {' '.join(hyper_params['seeds'])} \
                                --eval_combined \
            """

    run_job(
        job_name=f"test_ensembling",
        job_cmd=job_cmd,
        partition='gpu-2h',
        log_dir='./logs',
        num_jobs_in_array=num_jobs
    )
