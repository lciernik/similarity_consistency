import os
import json
from helper import load_models, get_hyperparams, parse_datasets
from slurm import run_job
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--models_config', type=str, default='./filtered_models_config.json')
parser.add_argument('--datasets', type=str, nargs='+', default=['wds/imagenet1k', 'wds/imagenetv2', 'wds/imagenet-a', 'wds/imagenet-r', 'wds/imagenet_sketch'],
                    help="datasets can be a list of dataset names or a file (e.g., webdatasets.txt) containing dataset names.")
parser.add_argument('--combination', type=str, default='ensemble', choices=['ensemble', 'concat', 'concat_pca'], 
                    help="Model combination to use")
parser.add_argument('--anchor_model', type=str, default='OpenCLIP_ViT-L-14_openai', help="Model combination to use")

args = parser.parse_args()

MODELS_CONFIG = args.models_config
DATASETS = " ".join(parse_datasets(args.datasets))

BASE_PROJECT_PATH = "/home/space/diverse_priors"
DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets')
FEATURES_ROOT = os.path.join(BASE_PROJECT_PATH, 'features')
MODELS_ROOT = os.path.join(BASE_PROJECT_PATH, 'models')
OUTPUT_ROOT = os.path.join(BASE_PROJECT_PATH, 'results')

if __name__ == "__main__":
    
    models, n_models = load_models(MODELS_CONFIG)
    
    assert args.anchor_model in models.keys(), f"Model in {args.anchor_model} not available in {MODELS_CONFIG=}."
    models.pop(args.anchor_model)

    if 'SegmentAnything_vit_b' in models.keys():
        models.pop('SegmentAnything_vit_b')

    model_keys = [sorted([args.anchor_model, val]) for val in models.keys()]

    # Extracting hyperparameters for evaluation: learning rate, few-shot k samples, epoch numbers, and seeds.
    hyper_params, num_jobs = get_hyperparams(num_seeds=3, size='imagenet1k')

    val_proportion = 0.2

    print("We evaluate the following hyperparameter", hyper_params)

    # Run evaluation for each model set
    for model_set in model_keys:
        print(f"Submitting Job with model_key{' '.join(model_set)}")
        job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
                export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
                clip_benchmark --dataset {DATASETS} \
                               --dataset_root {DATASETS_ROOT} \
                               --feature_root {FEATURES_ROOT} \
                               --model_root {MODELS_ROOT} \
                               --output_root {OUTPUT_ROOT} \
                               --models_config_file {MODELS_CONFIG} \
                               --task=linear_probe \
                               --mode={'ensemble' if args.combination == 'ensemble' else 'combined_models'} \
                               --feature_combiner {args.combination if args.combination != 'ensemble' else 'concat'} \
                               --model_key {' '.join(model_set)} \
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
        mem = 32 if args.combination == 'ensemble' else 128

        run_job(
            job_name=f"combined_eval",
            job_cmd=job_cmd,
            partition='cpu-5h' if args.combination == 'ensemble' else 'gpu-2d',
            log_dir=f'{OUTPUT_ROOT}/logs',
            num_jobs_in_array=num_jobs,
            mem=mem
        )
