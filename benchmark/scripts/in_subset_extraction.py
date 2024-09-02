import os

from helper import load_models
from slurm import run_job

MODELS_CONFIG = "./new_models_config.json"

PROJECT_PATH = "/home/space/diverse_priors"

FEATURES_ROOT = os.path.join(PROJECT_PATH, 'features', 'wds_imagenet1k')
SUBSET_IDXS = os.path.join(PROJECT_PATH, 'datasets', 'imagenet-subset-{num_samples_class}k',
                           'imagenet-{num_samples_class}k-{split}.json')
OUTPUT_ROOT = os.path.join(PROJECT_PATH, 'features', 'imagenet-subset-{num_samples_class}k')

if __name__ == "__main__":

    models, n_models = load_models(MODELS_CONFIG)

    model_keys = ' '.join(models.keys())

    # model_keys = "Kakaobrain_Align"

    # nr_samples = [1, 5, 10, 20, 30, 40]
    nr_samples = [10]

    for num_samples_class in nr_samples:
        for split in ['train', 'test']:
            job_cmd = f"""
            python feature_extraction_imagenet_subset.py \
                    --features_root {FEATURES_ROOT} \
                    --model_key {model_keys} \
                    --split {split} \
                    --num_samples_class {num_samples_class} \
                    --subset_idxs {SUBSET_IDXS} \
                    --output_root_dir {OUTPUT_ROOT}
            """

            run_job(
                job_name=f"feat_extr_in_sub_{num_samples_class}_{split}",
                job_cmd=job_cmd,
                partition='cpu-2h',
                log_dir='./logs',
                num_jobs_in_array=1,
                mem=64
            )
