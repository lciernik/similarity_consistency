import os
import random
import sys
from datetime import datetime
from itertools import product

sys.path.append('..')
from slurm import run_job
from helper import load_models, get_hyperparams

MODELS_CONFIG = "../models_config.json"
# DATASETS = "./test_webdatasetst.txt"
DATASETS = "imagenet-subset-10k"
DATASETS_ROOT = "/home/space/diverse_priors/datasets"

# Create new test experiment folder
BASE_PATH_EXP = "./test_results"
current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M")
BASE_PATH_EXP = os.path.join(BASE_PATH_EXP, current_datetime)
os.makedirs(BASE_PATH_EXP, exist_ok=True)

FEATURES_ROOT = os.path.join(BASE_PATH_EXP, 'features')
OUTPUT_ROOT = os.path.join(BASE_PATH_EXP, 'results')
MODEL_ROOT = os.path.join(BASE_PATH_EXP, 'models')
os.makedirs(FEATURES_ROOT, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)

if __name__ == "__main__":
    # Select random model
    # models, n_models = load_models(MODELS_CONFIG)
    # random_model = random.choice(list(models.keys()))

    # Select random hyperparameters
    # hyper_params, _ = get_hyperparams(num_seeds=1, size='small')
    # hyper_params = {k: [random.choice(v)] for k, v in hyper_params.items()}
    

    # Set previous run hyperparameters
    random_model="vgg16 vgg19 seresnet50 resnet50 resnet152"

    hyper_params = dict(
            fewshot_lrs=['0.01'],
            fewshot_ks=['5'],
            fewshot_epochs=['50'],
            seeds=['0'],
        )
    
    num_jobs = len(list(product(*hyper_params.values())))

    val_proportion = 0

    print(f"Testing single model {random_model} with hyper_params: {hyper_params}")

    # Evaluate each model on all data and all hyperparameter configurations.
    job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
    clip_benchmark --dataset={DATASETS} \
                   --dataset_root={DATASETS_ROOT} \
                   --feature_root={FEATURES_ROOT} \
                   --output_root={OUTPUT_ROOT} \
                   --model_root={MODEL_ROOT} \
                   --task=linear_probe \
                   --mode=single_model \
                   --model_key {random_model} \
                   --models_config_file={MODELS_CONFIG} \
                   --batch_size=64 \
                   --fewshot_k {' '.join(hyper_params['fewshot_ks'])} \
                   --fewshot_lr {' '.join(hyper_params['fewshot_lrs'])} \
                   --fewshot_epochs {' '.join(hyper_params['fewshot_epochs'])} \
                   --train_split train \
                   --test_split test \
                   --val_proportion={val_proportion} \
                   --seed {' '.join(hyper_params['seeds'])} 
    """

    run_job(
        job_name=f"test_single",
        job_cmd=job_cmd,
        partition='gpu-2h',
        log_dir=f'{BASE_PATH_EXP}/logs',
        num_jobs_in_array=num_jobs
    )
