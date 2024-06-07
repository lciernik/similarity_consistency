import os
import json
from slurm import run_job
from helper import load_models
from itertools import product


if __name__ == "__main__":

    model_key="Kakaobrain_Align"

    nr_samples = [10, 20, 30, 40, 50, 80, 160]

    for num_samples_class in nr_samples:
        for split in ['train', 'test']:
            job_cmd = f"python feature_extraction_imagenet_subset.py --num_samples_class={num_samples_class} --split={split}" # Run for all models.
            # job_cmd = f"python feature_extraction_imagenet_subset.py --num_samples_class={num_samples_class} --split={split} --model_key {model_key}"

            run_job(
                job_name=f"feat_extr_in_sub_{num_samples_class}_{split}",
                job_cmd=job_cmd,
                partition='cpu-2h',
                log_dir='./logs',
                num_jobs_in_array=1
            )
