import os
import json
from slurm import run_job
from helper import load_models, prepare_for_combined_usage, count_nr_datasets

MODELS_CONFIG = "./models_config.json"

BASE_PROJECT_PATH = "/home/space/diverse_priors"

# DATASETS = "./webdatasets.txt"
# DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets', 'wds', 'wds_{dataset_cleaned}')

DATASETS = "./imagenet_subset.txt"
DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets')

FEATURES_ROOT = os.path.join(BASE_PROJECT_PATH, 'features')

SIM_METHOD = 'rsa'  # Distance matrix computation method
CORR_METHOD = 'spearman'

private_out_root = f"/home/lciernik/projects/divers-priors/results_local/{SIM_METHOD}_correlation_{CORR_METHOD}"
OUTPUT_ROOT = os.path.join(private_out_root, 'imagenet_subset_10k')

if __name__ == "__main__":
    # Retrieve the configuration of all models we intend to evaluate.
    models, n_models = load_models(MODELS_CONFIG)
    print(f"Run CKA distance matrix experiment with {n_models} models.")
    # Prepare the models for combined usage.
    model_names, sources, model_parameters, module_names = prepare_for_combined_usage(models)

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
                                            --model {' '.join(model_names)} \
                                            --model_source {' '.join(sources)} \
                                            --model_parameters {' '.join([f"'{json.dumps(x)}'" for x in model_parameters])} \
                                            --module_name {' '.join(module_names)} \
                                            --train_split train \
                                            --sim_method {SIM_METHOD} \
                                            --corr_method {CORR_METHOD}
                    """
    run_job(
        job_name=f"CKA",
        job_cmd=job_cmd,
        partition='gpu-2d',
        log_dir='./logs',
        num_jobs_in_array=njobs,
    )
