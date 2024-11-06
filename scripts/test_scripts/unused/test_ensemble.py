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
DATASETS = "imagenet-subset-10k"  # "./test_webdatasetst.txt"
DATASETS_ROOT = "/home/space/diverse_priors/datasets"

BASE_PATH_EXP = "/home/lciernik/projects/divers-priors/diverse_priors/benchmark/scripts/test_scripts/test_results/2024_05_26_17_26" # Path to a previous run with single model evaluation for vgg16 vgg19 seresnet50 and resnet50 resnet152

FEATURES_ROOT = os.path.join(BASE_PATH_EXP, 'features')
MODEL_ROOT = os.path.join(BASE_PATH_EXP, 'models')
OUTPUT_ROOT = os.path.join(BASE_PATH_EXP, 'results')

if __name__ == "__main__":
    # Two random models to evaluate in combination
    # models, n_models = load_models(MODELS_CONFIG)
    # models2combine = random.sample(list(models.keys()), 2)
    # models = {k: v for k, v in models.items() if k in models2combine}

    # models2combine = "vgg16 vgg19 seresnet50"
    models2combine = "resnet50 resnet152"

    # Select random hyperparameters
    # hyper_params, _ = get_hyperparams(num_seeds=10)
    # hyper_params = {k: [random.choice(v)] for k, v in hyper_params.items()}

    hyper_params = dict(
            fewshot_lrs=['0.01'],
            fewshot_ks=['5'],
            fewshot_epochs=['50'],
            seeds=['0'],
        )

    num_jobs = len(list(product(*hyper_params.values())))

    print(f"Testing ensembling models {models2combine} with hyper_params {hyper_params}")

    job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
            export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
            clip_benchmark --dataset={DATASETS} \
                           --dataset_root={DATASETS_ROOT} \
                           --feature_root={FEATURES_ROOT} \
                           --output_root={OUTPUT_ROOT} \
                           --model_root={MODEL_ROOT} \
                           --task=linear_probe \
                           --mode=ensemble \
                           --model_key {models2combine} \
                           --models_config_file={MODELS_CONFIG} \
                           --batch_size=64 \
                           --fewshot_k {' '.join(hyper_params['fewshot_ks'])} \
                           --fewshot_lr {' '.join(hyper_params['fewshot_lrs'])} \
                           --fewshot_epochs {' '.join(hyper_params['fewshot_epochs'])} \
                           --train_split train \
                           --test_split test \
                           --seed {' '.join(hyper_params['seeds'])}
            """

    run_job(
        job_name=f"test_ensembling",
        job_cmd=job_cmd,
        partition='gpu-2h',
        log_dir=f'{BASE_PATH_EXP}/logs',
        num_jobs_in_array=num_jobs
    )
