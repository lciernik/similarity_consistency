import os

from helper import load_models
from project_location import FEATURES_ROOT as feat_root
from project_location import SUBSET_ROOT, DATASETS_ROOT
from slurm import run_job

# MODELS_CONFIG = "./configs/models_config_wo_alignment.json"
MODELS_CONFIG = "./configs/models_config_anchor_models.json"

FEATURES_ROOT = os.path.join(feat_root, 'wds_imagenet1k')
SUBSET_IDXS = os.path.join(DATASETS_ROOT, 'imagenet-subset-{num_samples_class}k-bootstrap', 'imagenet-{num_samples_class}k-{split}-seed-{seed}.json')
OUTPUT_ROOT = '/home/space/diverse_priors/features_bootstrap/imagenet-subset-{num_samples_class}k-seed-{seed}'

if __name__ == "__main__":

    models, n_models = load_models(MODELS_CONFIG)

    model_keys = ' '.join(models.keys())

    
    num_samples_class=30
    split='train'
    max_seed=500

    # for seed in range(3):
    #     curr_subset_idxs = SUBSET_IDXS.format(num_samples_class=num_samples_class, split=split, seed=seed)
    #     curr_output_root = OUTPUT_ROOT.format(num_samples_class=num_samples_class, split=split, seed=seed)
        
    #     job_cmd = f"""
    #     python feature_extraction_imagenet_subset_bt.py \
    #             --features_root {FEATURES_ROOT} \
    #             --model_key {model_keys} \
    #             --split {split} \
    #             --num_samples_class {num_samples_class} \
    #             --subset_idxs {curr_subset_idxs} \
    #             --output_root_dir {curr_output_root} 
    #     """

    #     run_job(
    #         job_name=f"feat_extr_in_sub_10_train_seed_{seed:03d}",
    #         job_cmd=job_cmd,
    #         partition='cpu-2h',
    #         log_dir='./logs',
    #         num_jobs_in_array=1,
    #         mem=64
    #     )

    for model_key in models.keys():
            
            job_cmd = f"""
            python feature_extraction_imagenet_subset_bt.py \
                    --features_root {FEATURES_ROOT} \
                    --model_key {model_key} \
                    --split {split} \
                    --num_samples_class {num_samples_class} \
                    --subset_idxs {SUBSET_IDXS} \
                    --output_root_dir {OUTPUT_ROOT} \
                    --max_seed {max_seed}
            """

            run_job(
                job_name=f"feat_extr_in_sub_10_train",
                job_cmd=job_cmd,
                partition='cpu-2h',
                log_dir='./logs',
                num_jobs_in_array=1,
                mem=64
            )
