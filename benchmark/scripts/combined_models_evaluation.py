import os
import json
from helper import load_models, get_hyperparams
from slurm import run_job
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--models_config', type=str, default='./models_config.json')
parser.add_argument('--datasets', type=str, nargs='+', default=['wds/imagenet1k'])
parser.add_argument('--sampling_folder', type=str, nargs='+')
parser.add_argument('--combination', type=str, default='ensemble')

args = parser.parse_args()

MODELS_CONFIG = args.models_config
# MODELS_CONFIG = "./models_config.json"
# DATASETS = "./webdatasets.txt"
DATASETS = " ".join(args.datasets)

BASE_PROJECT_PATH = "/home/space/diverse_priors"
DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets')
FEATURES_ROOT = os.path.join(BASE_PROJECT_PATH, 'features')
MODELS_ROOT = os.path.join(BASE_PROJECT_PATH, 'models')
OUTPUT_ROOT = os.path.join(BASE_PROJECT_PATH, 'results')
SAMPLING_ROOT = os.path.join(BASE_PROJECT_PATH, 'sampling')

if __name__ == "__main__":
    # Retrieve the configuration of all models we intend to evaluate.
    # models, n_models = load_models(MODELS_CONFIG)
    # model_keys = ' '.join(models.keys())

    # We find all files in each sampling folder and load the jsons
    model_keys = []
    for sampling_folder in args.sampling_folder:
        cur_folder = os.path.join(SAMPLING_ROOT, sampling_folder)
        for file in os.listdir(cur_folder):
            if file.endswith(".json"):
                # Open the JSON, which should contain a list of lists
                with open(os.path.join(cur_folder, file), 'r') as f:
                    model_keys.extend(json.load(f))

    # Sort each list within model_keys alphabetically
    model_keys = [sorted(model_key) for model_key in model_keys]

    # Filter out duplicates
    model_keys = list(set([tuple(model_key) for model_key in model_keys]))

    print("Running evaluation for the following model sets:")
    print("\n".join([str(model_key) for model_key in model_keys]))

    # Extracting hyperparameters for evaluation: learning rate, few-shot k samples, epoch numbers, and seeds.
    hyper_params, num_jobs = get_hyperparams(num_seeds=5, size='imagenet1k')

    val_proportion = 0

    # Run evaluation for each model set
    for model_set in model_keys[:2]:  # TODO only two runs for testing!
        job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
                export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
                clip_benchmark --dataset {DATASETS} \
                               --dataset_root {DATASETS_ROOT} \
                               --feature_root {FEATURES_ROOT} \
                               --model_root {MODELS_ROOT} \
                               --output_root {OUTPUT_ROOT} \
                                --models_config_file {MODELS_CONFIG} \
                               --task=linear_probe \
                               --mode=combined_models
                               --feature_combiner {args.combination} \
                               --model_key {' '.join(model_set)} \
                               --models_config_file {MODELS_CONFIG} \
                               --batch_size=64 \
                               --fewshot_k {' '.join(hyper_params['fewshot_ks'])} \
                               --fewshot_lr {' '.join(hyper_params['fewshot_lrs'])} \
                               --fewshot_epochs {' '.join(hyper_params['fewshot_epochs'])} \
                               --train_split train \
                               --test_split test \
                               --val_proportion {val_proportion} \
                               --seed {' '.join(hyper_params['seeds'])} \
                """

        run_job(
            job_name=f"combined_eval",
            job_cmd=job_cmd,
            # Note: this code runs much longer compared to the feature extraction code! It iterates over all possible
            # model combinations.
            partition='cpu-2h' if args.combination == 'ensemble' else 'gpu-5h',
            log_dir=f'{OUTPUT_ROOT}/logs',
            num_jobs_in_array=num_jobs
        )
