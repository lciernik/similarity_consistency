import os
from slurm import run_job

MODELS_CONFIG = "./scripts/models_config.json"
BASE_PROJECT_PATH = "/home/space/diverse_priors"
OUTPUT_ROOT = os.path.join(BASE_PROJECT_PATH, 'sampling')

if __name__ == "__main__":
    num_samples = 10

    num_model_it = range(2, 7)
    for num_models in num_model_it:
        job_cmd = f"""
        python clip_benchmark/sample_models.py  \
            --num_samples {num_samples} \
            --num_models {num_models} \
            --sampling_strategies top-k random \
            --model_config_path {MODELS_CONFIG} \
            --output_root {OUTPUT_ROOT}/models_{num_models}-samples_{num_samples}
        """
        run_job(
            job_name=f"sampling",
            job_cmd=job_cmd,
            partition='cpu-9m',
            log_dir='./logs',
            num_jobs_in_array=1
        )
