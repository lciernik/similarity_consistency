import os
import sys
import json
from datetime import datetime
import random
from itertools import product

sys.path.append('..')
from slurm import run_job
from helper import load_models, get_hyperparams, prepare_for_combined_usage

MODELS_CONFIG = "../models_config.json"
# DATASETS = "./test_webdatasetst.txt"
DATASETS = "wds/vtab/cifar10"
BASE_PROJECT_PATH = "/home/space/diverse_priors"
DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets')
FEATURES_ROOT = os.path.join(BASE_PROJECT_PATH, 'features')

# Create new test experiment folder
BASE_PATH_EXP = "./test_results"
current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M")
BASE_PATH_EXP = os.path.join(BASE_PATH_EXP, current_datetime)
os.makedirs(BASE_PATH_EXP, exist_ok=True)

OUTPUT_ROOT = os.path.join(BASE_PATH_EXP, 'results')
MODELS_ROOT = os.path.join(BASE_PATH_EXP, 'models')
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(MODELS_ROOT, exist_ok=True)

COMBINERS = ["concat", "concat_pca"]

if __name__ == "__main__":
    # Two random models to evaluate in combination
    # models, n_models = load_models(MODELS_CONFIG)
    # models2combine = random.sample(list(models.keys()), 2)
    # models = {k: v for k, v in models.items() if k in models2combine}
    models = "dinov2-vit-large-p14 DreamSim_open_clip_vitb32"

    # Select random hyperparameters
    # hyper_params, _ = get_hyperparams(num_seeds=10)
    # hyper_params = {k: [random.choice(v)] for k, v in hyper_params.items()}
    # num_jobs = len(list(product(*hyper_params.values())))

    hyper_params = dict(
            fewshot_lrs=['0.1'],
            fewshot_ks=['-1'],
            fewshot_epochs=['20'],
            seeds=['0'],
        )

    val_proportion=0

    # random combiner
    # combiner = random.choice(COMBINERS)
    combiner = "concat"

    print(f"Testing combined models {models} with hyper_params {hyper_params} and feature combiner {combiner}")

    job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
            export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
            clip_benchmark --dataset {DATASETS} \
                           --dataset_root {DATASETS_ROOT} \
                           --feature_root {FEATURES_ROOT} \
                           --model_root {MODELS_ROOT} \
                           --output_root {OUTPUT_ROOT} \
                           --task=linear_probe \
                           --mode=combined_models \
                           --model_key {models} \
                           --models_config_file {MODELS_CONFIG} \
                           --batch_size=64 \
                           --fewshot_k {' '.join(hyper_params['fewshot_ks'])} \
                           --fewshot_lr {' '.join(hyper_params['fewshot_lrs'])} \
                           --fewshot_epochs {' '.join(hyper_params['fewshot_epochs'])} \
                           --train_split train \
                           --test_split test \
                           --val_proportion {val_proportion} \
                           --seed {' '.join(hyper_params['seeds'])} \
                           --feature_combiner {combiner}
            """

    run_job(
        job_name=f"test_combined",
        job_cmd=job_cmd,
        partition='gpu-2h',
        log_dir=f'{BASE_PATH_EXP}/logs',
        num_jobs_in_array=1
    )
