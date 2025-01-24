import os

from helper import load_models
from project_location import FEATURES_ROOT as feat_root
from project_location import SUBSET_ROOT
from slurm import run_job

# MODELS_CONFIG = "./configs/models_config_wo_alignment.json"
MODELS_CONFIG = "./configs/models_config_places_vs_in1k.json"

FEATURES_ROOT = os.path.join(feat_root, 'wds_imagenet1k')
SUBSET_IDXS = os.path.join(SUBSET_ROOT, 'imagenet-subset-{num_samples_class}k',
                           'imagenet-{num_samples_class}k-{split}.json')
OUTPUT_ROOT = os.path.join(feat_root, 'imagenet-subset-{num_samples_class}k')

if __name__ == "__main__":

    models, n_models = load_models(MODELS_CONFIG)

    model_keys = ' '.join(models.keys())

    nr_samples = [1, 5, 10, 20, 30, 40]

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
