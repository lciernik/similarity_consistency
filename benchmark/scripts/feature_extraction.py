import os

from helper import load_models, parse_datasets
from slurm import run_job
from project_location import DATASETS_ROOT, FEATURES_ROOT

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--models_config', type=str, default='./configs/models_config_wo_alignment.json')
parser.add_argument('--datasets', type=str, nargs='+', default='./configs/webdatasets_w_in1k.txt',
                    help="datasets can be a list of dataset names or a file (e.g., webdatasets.txt) containing dataset names.")
args = parser.parse_args()

MODELS_CONFIG = args.models_config

DATASETS = " ".join(parse_datasets(args.datasets))

if __name__ == "__main__":
    # Retrieve the configuration of all models we intend to evaluate.
    models, n_models = load_models(MODELS_CONFIG)

    model_keys = list(models.keys())

    # Extract features for all models and datasets.
    for key in model_keys:
        print(f"Running feature extraction for {key}")
        job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
        export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
        clip_benchmark --dataset {DATASETS} \
                       --dataset_root {DATASETS_ROOT} \
                       --feature_root {FEATURES_ROOT} \
                       --task=feature_extraction \
                       --model_key {key} \
                       --models_config_file {MODELS_CONFIG} \
                       --batch_size=64 \
                       --train_split train \
                       --test_split test \
                       --num_workers=0
        """

        run_job(
            job_name=f"feat_extr_{key}",
            job_cmd=job_cmd,
            partition='gpu-2d',
            log_dir=f'{FEATURES_ROOT}/logs',
            num_jobs_in_array=1,
            mem=64
        )
