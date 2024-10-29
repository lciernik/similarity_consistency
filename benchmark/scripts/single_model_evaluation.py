import os
import json
from slurm import run_job
from helper import load_models, get_hyperparams, parse_datasets
from project_location import DATASETS_ROOT, FEATURES_ROOT, MODELS_ROOT, RESULTS_ROOT
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

    # Extracting hyperparameters for evaluation: learning rate, few-shot k samples, epoch numbers, and seeds.
    hyper_params, num_jobs = get_hyperparams(num_seeds=3, size='imagenet1k')

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
                       --output_root {RESULTS_ROOT} \
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
                       --seed {' '.join(hyper_params['seeds'])} \
                       --force_train
        """

        run_job(
            job_name=f"probe_{key}",
            job_cmd=job_cmd,
            partition='gpu-2d',
            log_dir=f'{RESULTS_ROOT}/logs',
            num_jobs_in_array=num_jobs
        )
