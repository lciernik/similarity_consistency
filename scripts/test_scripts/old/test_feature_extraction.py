import os
import random
import sys
from datetime import datetime
from itertools import product

sys.path.append('..')
from slurm import run_job
from helper import load_models, get_hyperparams

MODELS_CONFIG = "../models_config.json"
DATASETS = "./test_webdatasetst.txt"
DATASETS_ROOT = "/home/space/diverse_priors/datasets"

# Create new test experiment folder
BASE_PATH_EXP = "./test_results"
current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M")
BASE_PATH_EXP = os.path.join(BASE_PATH_EXP, current_datetime)
os.makedirs(BASE_PATH_EXP, exist_ok=True)

FEATURES_ROOT = os.path.join(BASE_PATH_EXP, 'features')
os.makedirs(FEATURES_ROOT, exist_ok=True)

if __name__ == "__main__":
    # Select random model
    models, n_models = load_models(MODELS_CONFIG)
    random_model = random.choice(list(models.keys()))

    print(f"Testing feature extraction with {random_model}.")

    # Evaluate each model on all data and all hyperparameter configurations.
    job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
    clip_benchmark --dataset={DATASETS} \
                   --dataset_root={DATASETS_ROOT} \
                   --feature_root={FEATURES_ROOT} \
                   --task=feature_extraction \
                   --model_key={random_model} \
                   --models_config_file={MODELS_CONFIG} \
                   --batch_size=64 \
                   --train_split train \
                   --test_split test
    """

    run_job(
        job_name=f"test_single",
        job_cmd=job_cmd,
        partition='gpu-2h',
        log_dir=f'{BASE_PATH_EXP}/logs',
        num_jobs_in_array=1
    )
