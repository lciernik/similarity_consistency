import os
import json
from slurm import run_job
from helper import load_models, get_hyperparams, parse_datasets

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--models_config', type=str, default='./models_config.json')
parser.add_argument('--datasets', type=str, nargs='+', default=['wds/imagenet1k', 'wds/imagenetv2', 'wds/imagenet-a', 'wds/imagenet-r', 'wds/imagenet_sketch'],
                    help="datasets can be a list of dataset names or a file (e.g., webdatasets.txt) containing dataset names.")
args = parser.parse_args()

MODELS_CONFIG = args.models_config
DATASETS = " ".join(parse_datasets(args.datasets))

BASE_PROJECT_PATH = "/home/space/diverse_priors"
DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets')
FEATURES_ROOT = os.path.join(BASE_PROJECT_PATH, 'features')
MODELS_ROOT = os.path.join(BASE_PROJECT_PATH, 'models')
OUTPUT_ROOT = os.path.join(BASE_PROJECT_PATH, 'results')

if __name__ == "__main__":
    # Retrieve the configuration of all models we intend to evaluate.
    models, n_models = load_models(MODELS_CONFIG)
    if 'SegmentAnything_vit_b' in models.keys():
        models.pop('SegmentAnything_vit_b')

    # Extracting hyperparameters for evaluation: learning rate, few-shot k samples, epoch numbers, and seeds.
    hyper_params, num_jobs = get_hyperparams(num_seeds=5, size='imagenet1k')

    # With val_proportion 0 we do not optimize weight decay!
    val_proportion = 0.2

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
                       --batch_size=1024 \
                       --fewshot_k {' '.join(hyper_params['fewshot_ks'])} \
                       --fewshot_lr {' '.join(hyper_params['fewshot_lrs'])} \
                       --fewshot_epochs {' '.join(hyper_params['fewshot_epochs'])} \
                       --reg_lambda {hyper_params['reg_lambda']} \
                       --regularization {' '.join(hyper_params['regularization'])} \
                       --train_split train \
                       --test_split test \
                       --val_proportion {val_proportion} \
                       --seed {' '.join(hyper_params['seeds'])} 
        """

        run_job(
            job_name=f"probe_{key}",
            job_cmd=job_cmd,
            partition='gpu-2d',
            log_dir=f'{OUTPUT_ROOT}/logs',
            num_jobs_in_array=num_jobs
        )
