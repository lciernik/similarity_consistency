import argparse
import json

from helper import load_models, parse_datasets
from project_location import DATASETS_ROOT, SUBSET_ROOT, FEATURES_ROOT, MODEL_SIM_ROOT
from slurm import run_job

parser = argparse.ArgumentParser()
parser.add_argument('--models_config', type=str, default='./configs/models_config_wo_alignment.json')
parser.add_argument('--datasets', type=str, nargs='+', default='./configs/webdatasets_w_insub10k.txt',
                    help="datasets can be a list of dataset names or a file (e.g., webdatasets.txt) containing "
                         "dataset names.")
args = parser.parse_args()

MODELS_CONFIG = args.models_config
DATASETS = parse_datasets(args.datasets)

SIM_METRIC_CONFIG = "./configs/similarity_metric_config_local_global.json"
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
                      clip_benchmark --dataset {datasets} \
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
        partition = 'gpu-2d' if exp_dict['sim_method'] == 'cka' else 'cpu-2d'
        mem = 150

        run_job(
            job_name=f"{exp_dict['sim_method'].capitalize()}",
            job_cmd=job_cmd,
            partition=partition,
            log_dir=f'{MODEL_SIM_ROOT}/logs',
            num_jobs_in_array=num_jobs,
            mem=mem
        )
