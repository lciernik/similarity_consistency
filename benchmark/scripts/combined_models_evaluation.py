import os

from helper import load_models, get_hyperparams
from slurm import run_job

MODELS_CONFIG = "./models_config.json"
DATASETS = "./webdatasets.txt"

BASE_PROJECT_PATH = "/home/space/diverse_priors"
DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets')
FEATURES_ROOT = os.path.join(BASE_PROJECT_PATH, 'features')
MODELS_ROOT = os.path.join(BASE_PROJECT_PATH, 'models')
OUTPUT_ROOT = os.path.join(BASE_PROJECT_PATH, 'results')

COMBINERS = ["concat", "concat_pca"]

if __name__ == "__main__":
    # Retrieve the configuration of all models we intend to evaluate.
    models, n_models = load_models(MODELS_CONFIG)
    model_keys = ' '.join(models.keys())

    # Extracting hyperparameters for evaluation: learning rate, few-shot k samples, epoch numbers, and seeds.
    hyper_params, num_jobs = get_hyperparams(num_seeds=10)

    val_proportion = 0.2

    for combiner in COMBINERS:
        job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
                export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
                clip_benchmark --dataset {DATASETS} \
                               --dataset_root {DATASETS_ROOT} \
                               --feature_root {FEATURES_ROOT} \
                               --model_root {MODELS_ROOT} \
                               --output_root {OUTPUT_ROOT} \
                               --task=linear_probe \
                               --mode=combined_models \
                               --model_key {model_keys} \
                               --models_config_file {MODELS_CONFIG} \
                               --batch_size=64 \
                               --fewshot_k {' '.join(hyper_params['fewshot_ks'])} \
                               --fewshot_lr {' '.join(hyper_params['fewshot_lrs'])} \
                               --fewshot_epochs {' '.join(hyper_params['fewshot_epochs'])} \
                               --train_split train \
                               --test_split test \
                               --val_proportion \
                               --seed {' '.join(hyper_params['seeds'])} \
                               --feature_combiner {combiner}
                """

        run_job(
            job_name=f"combined_eval",
            job_cmd=job_cmd,
            # Note: this code runs much longer compared to the feature extraction code! It iterates over all possible
            # model combinations.
            partition='gpu-2d',
            log_dir=f'{OUTPUT_ROOT}/logs',
            num_jobs_in_array=num_jobs
        )