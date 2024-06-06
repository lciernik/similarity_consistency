import os
import json
from slurm import run_job
from helper import load_models, prepare_for_combined_usage, count_nr_datasets

MODELS_CONFIG = "./models_config.json"

BASE_PROJECT_PATH = "/home/space/diverse_priors"

# DATASETS = ["imagenet-subset-10k", "imagenet-subset-20k", "imagenet-subset-40k", "imagenet-subset-80k", "imagenet-subset-160k"]
DATASETS = ["imagenet-subset-10k", "imagenet-subset-20k", "imagenet-subset-30k"]
DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets')
FEATURES_ROOT = os.path.join(BASE_PROJECT_PATH, 'features')
MODELS_ROOT = os.path.join(BASE_PROJECT_PATH, 'models')
OUTPUT_ROOT = os.path.join(BASE_PROJECT_PATH, 'model_similarities')

sim_method_config = [
    {
        'sim_method' : 'cka',
        'sim_kernel' : 'linear',
        'rsa_method' : 'correlation',
        'corr_method': 'pearson',
        'sigma': 0,
    },
    {
        'sim_method' : 'cka',
        'sim_kernel' : 'rbf',
        'rsa_method' : 'correlation',
        'corr_method': 'pearson',
        'sigma': 0.2,
    },
    {
        'sim_method' : 'cka',
        'sim_kernel' : 'rbf',
        'rsa_method' : 'correlation',
        'corr_method': 'pearson',
        'sigma': 0.4,
    },
    {
        'sim_method' : 'cka',
        'sim_kernel' : 'rbf',
        'rsa_method' : 'correlation',
        'corr_method': 'pearson',
        'sigma': 0.6,
    },
    {
        'sim_method' : 'cka',
        'sim_kernel' : 'rbf',
        'rsa_method' : 'correlation',
        'corr_method': 'pearson',
        'sigma': 0.8,
    },
    {
        'sim_method' : 'rsa',
        'sim_kernel' : 'linear',
        'rsa_method' : 'correlation',
        'corr_method': 'pearson',
        'sigma': 0,
    },
    {
        'sim_method' : 'rsa',
        'sim_kernel' : 'linear',
        'rsa_method' : 'correlation',
        'corr_method': 'spearman',
        'sigma': 0,
    },

]

if __name__ == "__main__":
    # Retrieve the configuration of all models we intend to evaluate.
    models, n_models = load_models(MODELS_CONFIG)
    models.pop('vgg16_gLocal')
    models.pop('Kakaobrain_Align')
    model_keys = ' '.join(models.keys())
    print(model_keys)

    num_jobs = len(DATASETS)

    datasets = ' '.join(DATASETS)
    
    for exp_dict in sim_method_config:
        print(f"Computing model similarity matrix with config:\n{json.dumps(exp_dict, indent=4)}")

        max_workers = 4

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
                                     --sigma {exp_dict['sigma']} 
                                     --max_workers {max_workers} \
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
