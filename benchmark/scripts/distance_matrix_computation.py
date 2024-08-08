import argparse
import json
import os

from helper import load_models, parse_datasets
from slurm import run_job

parser = argparse.ArgumentParser()
parser.add_argument('--models_config', type=str, default='./filtered_models_config.json')
parser.add_argument('--datasets', type=str, nargs='+', default='./webdatasets_wo_imagenet.txt',
                    help="datasets can be a list of dataset names or a file (e.g., webdatasets.txt) containing "
                         "dataset names.")
args = parser.parse_args()

MODELS_CONFIG = args.models_config

DATASETS = parse_datasets(args.datasets)

BASE_PROJECT_PATH = "/home/space/diverse_priors"
DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets')
SUBSET_ROOT = os.path.join(DATASETS_ROOT, 'subsets')
FEATURES_ROOT = os.path.join(BASE_PROJECT_PATH, 'features')
MODELS_ROOT = os.path.join(BASE_PROJECT_PATH, 'models')
OUTPUT_ROOT = os.path.join(BASE_PROJECT_PATH, 'model_similarities')

SIM_METRIC_CONFIG = "./similarity_metric_config.json"
with open(SIM_METRIC_CONFIG, "r") as file:
    sim_method_config = json.load(file)

if __name__ == "__main__":
    # Retrieve the configuration of all models we intend to evaluate.
    models, n_models = load_models(MODELS_CONFIG)
    if 'SegmentAnything_vit_b' in models.keys():
        models.pop('SegmentAnything_vit_b')
    model_keys = ' '.join(models.keys())

    num_jobs = len(DATASETS)

    datasets = ' '.join(DATASETS)

    for exp_dict in sim_method_config:
        print(f"Computing model similarity matrix with config:\n{json.dumps(exp_dict, indent=4)}")

        max_workers = 8

        job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
                      export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
                      clip_benchmark --dataset {datasets} \
                                     --dataset_root {DATASETS_ROOT} \
                                     --feature_root {FEATURES_ROOT} \
                                     --output {OUTPUT_ROOT} \
                                     --task=model_similarity \
                                     --model_key {model_keys} \
                                     --models_config_file {MODELS_CONFIG} \
                                     --train_split train \
                                     --sim_method {exp_dict['sim_method']} \
                                     --sim_kernel {exp_dict['sim_kernel']} \
                                     --rsa_method {exp_dict['rsa_method']} \
                                     --corr_method {exp_dict['corr_method']} \
                                     --sigma {exp_dict['sigma']} \
                                     --max_workers {max_workers} \
                                     --use_ds_subset \
                                     --subset_root {SUBSET_ROOT}
                        """
        partition = 'gpu-5h' if exp_dict['sim_method'] == 'cka' else 'cpu-2d'
        mem = 150

        run_job(
            job_name=f"{exp_dict['sim_method'].capitalize()}",
            job_cmd=job_cmd,
            partition=partition,
            log_dir=f'{OUTPUT_ROOT}/logs',
            num_jobs_in_array=num_jobs,
            mem=mem
        )
