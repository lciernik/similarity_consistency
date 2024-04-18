import os
import json
from slurm import run_job
from helper import load_models

MODELS_CONFIG = "./models_config_small.json"
DATASETS = "wds/voc2007"
# DATASETS = "./imagenet_subset.txt"

BASE_PROJECT_PATH = "/home/space/diverse_priors"
DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets', 'wds', 'wds_{dataset_cleaned}')
# DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets')

FEATURES_ROOT = os.path.join(BASE_PROJECT_PATH, 'features')

private_out_root = "/home/lciernik/projects/divers-priors/results_local/cka"
OUTPUT_ROOT = os.path.join(private_out_root, )

if __name__ == "__main__":
    # Retrieve the configuration of all models we intend to evaluate.
    models, n_models = load_models(MODELS_CONFIG)

    # Evaluate each model on all datasets and all hyperparameter configurations.
    for key, model_config in models.items():
        job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
        export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
        clip_benchmark eval --dataset {DATASETS} \
                            --dataset_root {DATASETS_ROOT} \
                            --feature_root {FEATURES_ROOT} \
                            --output {OUTPUT_ROOT} \
                            --task=model_similarity \
                            --model {model_config['model_name']} \
                            --model_source {model_config['source']} \
                            --model_parameters '{json.dumps(model_config['model_parameters'])}' \
                            --module_name {model_config['module_name']} \
                            --train_split train
        """

        run_job(
            job_name=f"feat_extr_{key}",
            job_cmd=job_cmd,
            partition='gpu-5h',
            log_dir='./logs',
            num_jobs_in_array=1
        )
