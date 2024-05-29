import os

from helper import load_models
from slurm import run_job

MODELS_CONFIG = "./models_config.json"
# DATASETS = "./webdatasets.txt" all datasets that we have
DATASETS = "imagenet-subset-10k"

BASE_PROJECT_PATH = "/home/space/diverse_priors"
DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets')
FEATURES_ROOT = os.path.join(BASE_PROJECT_PATH, 'features')

if __name__ == "__main__":
    # Retrieve the configuration of all models we intend to evaluate.
    models, n_models = load_models(MODELS_CONFIG)

    # Extract features for all models and datasets.
    for key, _ in models.items():
        job_cmd = f"""export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
        export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
        clip_benchmark --dataset {DATASETS} \
                       --dataset_root {DATASETS_ROOT} \
                       --feature_root {FEATURES_ROOT} \
                       --task=feature_extraction \
                       --model_key {key} \
                       --models_config_file {MODELS_CONFIG} \
                       --batch_size=64 \
                       --train_split train \
                       --test_split test
        """

        run_job(
            job_name=f"feat_extr_{key}",
            job_cmd=job_cmd,
            partition='gpu-5h',
            log_dir=f'{FEATURES_ROOT}/logs',
            num_jobs_in_array=1
        )
