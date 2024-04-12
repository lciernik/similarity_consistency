import os
import json
from slurm import run_job
from helper import load_models, get_hyperparams

MODELS_CONFIG = "./models_config.json"
DATASETS = "./webdatasets.txt"

BASE_PROJECT_PATH = "/home/space/diverse_priors"
DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets', 'wds', 'wds_{dataset_cleaned}')

FEATURES_ROOT = os.path.join(BASE_PROJECT_PATH, 'features')

OUTPUT_ROOT = os.path.join(BASE_PROJECT_PATH, 'results', 'single_models', '{fewshot_k}', '{dataset}', '{model}',
                           'fewshot_lr_{fewshot_lr}', 'fewshot_epochs_{fewshot_epochs}', 'seed_{seed}')

if __name__ == "__main__":
    # Retrieve the configuration of all models we intend to evaluate.
    models, n_models = load_models(MODELS_CONFIG)

    # Extracting hyperparameters for evaluation: learning rate, few-shot k samples, epoch numbers, and seeds.
    hyper_params, num_jobs = get_hyperparams(num_seeds=10)

    # Evaluate each model on all datasets and all hyperparameter configurations.
    for key, model_config in models.items():
        job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
        export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
        clip_benchmark eval --dataset {DATASETS} \
                            --dataset_root {DATASETS_ROOT} \
                            --feature_root {FEATURES_ROOT} \
                            --output {OUTPUT_ROOT} \
                            --task=linear_probe \
                            --model {model_config['model_name']} \
                            --model_source {model_config['source']} \
                            --model_parameters '{json.dumps(model_config['model_parameters'])}' \
                            --module_name {model_config['module_name']} \
                            --batch_size=64 \
                            --fewshot_k {' '.join(hyper_params['fewshot_ks'])} \
                            --fewshot_lr {' '.join(hyper_params['fewshot_lrs'])} \
                            --fewshot_epochs {' '.join(hyper_params['fewshot_epochs'])} \
                            --train_split train \
                            --test_split test \
                            --seed {' '.join(hyper_params['seeds'])} 
        """

        run_job(
            job_name=f"feat_extr_{key}",
            job_cmd=job_cmd,
            partition='gpu-5h',
            log_dir='./logs',
            num_jobs_in_array=1
        )
