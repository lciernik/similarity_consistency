import os
import json
from slurm import run_job
from helper import load_models, prepare_for_combined_usage, count_nr_datasets

MODELS_CONFIG = "./models_config.json"

BASE_PROJECT_PATH = "/home/space/diverse_priors"

DATASETS = "imagenet-subset-10k"
DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets')

FEATURES_ROOT = os.path.join(BASE_PROJECT_PATH, 'features')
MODELS_ROOT = os.path.join(BASE_PROJECT_PATH, 'models')
OUTPUT_ROOT = os.path.join(BASE_PROJECT_PATH, 'model_similarities')

SIM_METHOD = 'cka'  # Distance matrix computation method
private_out_root = f"/home/lciernik/projects/divers-priors/results_local/{SIM_METHOD}"

# SIM_METHOD = 'rsa'  # Distance matrix computation method
# CORR_METHOD = 'spearman'
# private_out_root = f"/home/lciernik/projects/divers-priors/results_local/{SIM_METHOD}_correlation_{CORR_METHOD}"
# --corr_method {CORR_METHOD}

if __name__ == "__main__":
    # Retrieve the configuration of all models we intend to evaluate.
    models, n_models = load_models(MODELS_CONFIG)
    print(f"Run CKA distance matrix experiment with {n_models} models.")

    # Nr of jobs in the array is equal to the number of datasets.
    njobs = count_nr_datasets(DATASETS)
    print(f"Nr.jobs: {njobs}")

    job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
                        export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
                        clip_benchmark eval --dataset {DATASETS} \
                                            --dataset_root {DATASETS_ROOT} \
                                            --feature_root {FEATURES_ROOT} \
                                            --output {OUTPUT_ROOT} \
                                            --task=model_similarity \
                                            --model_key {models} \
                                            --models_config_file {MODELS_CONFIG} \
                                            --train_split train \
                                            --sim_method {SIM_METHOD} 
                    """
    run_job(
        job_name=f"CKA",
        job_cmd=job_cmd,
        partition='gpu-2d',
        log_dir='./logs',
        num_jobs_in_array=njobs,
    )
