import argparse
import json

from helper import load_models, parse_datasets
from project_location import DATASETS_ROOT, SUBSET_ROOT
from slurm import run_job

parser = argparse.ArgumentParser()
parser.add_argument('--models_config', type=str, default='./configs/models_config_anchor_models.json')
parser.add_argument('--max_seed', default=500, type=int,
                    help='Maximum seed for the subset indices.')
args = parser.parse_args()


FEATURES_ROOT = "/home/space/diverse_priors/features_bootstrap"
MODEL_SIM_ROOT = "/home/space/diverse_priors/model_similarities_bootstrap"

MODELS_CONFIG = args.models_config

DATASETS = [ f"imagenet-subset-30k-seed-{seed}" for seed in range(args.max_seed)]

# SIM_METRIC_CONFIG = "./configs/similarity_metric_config_local_global.json"
SIM_METRIC_CONFIG = "./configs/similarity_metric_config_rbf_local.json"

with open(SIM_METRIC_CONFIG, "r") as file:
    sim_method_config = json.load(file)

if __name__ == "__main__":
    # Retrieve the configuration of all models we intend to evaluate.
    models, n_models = load_models(MODELS_CONFIG)

    model_keys = ' '.join(models.keys())

    num_jobs = len(DATASETS)

    datasets = ' '.join(DATASETS)

    for exp_dict in sim_method_config:
        print(f"Computing model similarity matrix with config:\n{json.dumps(exp_dict, indent=4)}")

        max_workers = 8

        job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
                      export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
                      sim_consistency --dataset {datasets} \
                                      --dataset_root {DATASETS_ROOT} \
                                      --feature_root {FEATURES_ROOT} \
                                      --output {MODEL_SIM_ROOT} \
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
        partition = 'gpu-9m' if exp_dict['sim_method'] == 'cka' else 'cpu-2h'
        mem = 64

        run_job(
            job_name=f"{exp_dict['sim_method'].capitalize()}",
            job_cmd=job_cmd,
            partition=partition,
            log_dir=f'{MODEL_SIM_ROOT}/logs',
            num_jobs_in_array=num_jobs,
            mem=mem
        )
