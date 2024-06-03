import os
import json
from slurm import run_job
from helper import load_models, get_hyperparams

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--models_config', type=str, default='./models_config.json')
args = parser.parse_args()

MODELS_CONFIG = args.models_config
# MODELS_CONFIG = "./models_config.json"
# DATASETS = "./webdatasets.txt"
DATASETS = "wds/imagenet1k"  # "imagenet-subset-10k"

BASE_PROJECT_PATH = "/home/space/diverse_priors"
DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets')
FEATURES_ROOT = os.path.join(BASE_PROJECT_PATH, 'features')
MODELS_ROOT = os.path.join(BASE_PROJECT_PATH, 'models')
OUTPUT_ROOT = os.path.join(BASE_PROJECT_PATH, 'results')

if __name__ == "__main__":
    # Retrieve the configuration of all models we intend to evaluate.
    models, n_models = load_models(MODELS_CONFIG)

    # Extracting hyperparameters for evaluation: learning rate, few-shot k samples, epoch numbers, and seeds.
    hyper_params, num_jobs = get_hyperparams(num_seeds=5, size='imagenet1k')

    # With val_proportion 0 we do not optimize weight decay!
    val_proportion = 0

    # Evaluate
    for i, (key, _) in enumerate(models.items()):
        job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
        export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
        clip_benchmark --dataset {DATASETS} \
                       --dataset_root {DATASETS_ROOT} \
                       --feature_root {FEATURES_ROOT} \
                       --model_root {MODELS_ROOT} \
                       --output_root {OUTPUT_ROOT} \
                       --task=linear_probe \
                       --mode=single_model \
                       --model_key {key} \
                       --models_config_file {MODELS_CONFIG} \
                       --batch_size=64 \
                       --fewshot_k {' '.join(hyper_params['fewshot_ks'])} \
                       --fewshot_lr {' '.join(hyper_params['fewshot_lrs'])} \
                       --fewshot_epochs {' '.join(hyper_params['fewshot_epochs'])} \
                       --train_split train \
                       --test_split test \
                       --val_proportion {val_proportion} \
                       --seed {' '.join(hyper_params['seeds'])} 
        """

        run_job(
            job_name=f"feat_extr_{key}",
            job_cmd=job_cmd,
            partition='gpu-2d',
            log_dir=f'{OUTPUT_ROOT}/logs',
            num_jobs_in_array=num_jobs
        )