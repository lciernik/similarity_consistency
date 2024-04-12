import os
import json
from slurm import run_job
from helper import load_models, get_hyperparams, prepare_for_combined_usage

MODELS_CONFIG = "./models_config.json"
DATASETS = "./webdatasets.txt"

BASE_PROJECT_PATH = "/home/space/diverse_priors"
DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets', 'wds', 'wds_{dataset_cleaned}')

FEATURES_ROOT = os.path.join(BASE_PROJECT_PATH, 'features')

OUTPUT_ROOT = os.path.join(BASE_PROJECT_PATH, 'results', 'combined_models', '{fewshot_k}', '{dataset}', '{model}',
                           'fewshot_lr_{fewshot_lr}', 'fewshot_epochs_{fewshot_epochs}', 'seed_{seed}')

COMBINERS = ["concat", "concat_pca"]

if __name__ == "__main__":
    # Retrieve the configuration of all models we intend to evaluate.
    models, n_models = load_models(MODELS_CONFIG)
    # Prepare job command input
    model_names, sources, model_parameters, module_names = prepare_for_combined_usage(models)
    # Extracting hyperparameters for evaluation: learning rate, few-shot k samples, epoch numbers, and seeds.
    hyper_params, num_jobs = get_hyperparams(num_seeds=10)

    for combiner in COMBINERS:
        job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
                export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
                clip_benchmark eval --dataset {DATASETS} \
                                    --dataset_root {DATASETS_ROOT} \
                                    --feature_root {FEATURES_ROOT} \
                                    --output {OUTPUT_ROOT} \
                                    --task=linear_probe \
                                    --model {' '.join(model_names)} \
                                    --model_source {' '.join(sources)} \
                                    --model_parameters {' '.join([f"'{json.dumps(x)}'" for x in model_parameters])} \
                                    --module_name {' '.join(module_names)} \
                                    --batch_size=64 \
                                    --fewshot_k {' '.join(hyper_params['fewshot_ks'])} \
                                    --fewshot_lr {' '.join(hyper_params['fewshot_lrs'])} \
                                    --fewshot_epochs {' '.join(hyper_params['fewshot_epochs'])} \
                                    --train_split train \
                                    --test_split test \
                                    --seed {' '.join(hyper_params['seeds'])} \
                                    --eval_combined \
                                    --feature_combiner {combiner}
                """

        run_job(
            job_name=f"combined_eval",
            job_cmd=job_cmd,
            # Note: this code runs much longer compared to the feature extraction code! It iterates over all possible
            # model combinations.
            partition='gpu-2d',
            log_dir='./logs',
            num_jobs_in_array=num_jobs
        )
