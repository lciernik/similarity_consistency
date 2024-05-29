import os
from slurm import run_job

MODELS_CONFIG = "./scripts/models_config.json"
BASE_PROJECT_PATH = "/home/space/diverse_priors"
OUTPUT_ROOT = os.path.join(BASE_PROJECT_PATH, 'sampling')

cluster_methods = [
    "cka_kernel_linear_unbiased",
    "cka_kernel_rbf_unbiased_sigma_0.4",
    "cka_kernel_rbf_unbiased_sigma_0.8",
    "rsa_method_correlation_corr_method_spearman",
    "cka_kernel_rbf_unbiased_sigma_0.2",
    "cka_kernel_rbf_unbiased_sigma_0.6",
    "rsa_method_correlation_corr_method_pearson"
]

if __name__ == "__main__":
    num_samples = 10

    num_model_it = range(2, 10)
    for num_models in num_model_it:
        # TODO change selection dataset to full imagenet as soon as other results are ready
        job_cmd = f"""python clip_benchmark/sample_models.py  \
            --num_samples {num_samples} \
            --num_models {num_models} \
            --sampling_strategies top-k random \
            --model_config_path {MODELS_CONFIG} \
            --selection_dataset imagenet-subset-10k \
            --output_root {OUTPUT_ROOT}/models_{num_models}-samples_{num_samples}"""

        run_job(
            job_name=f"sampling",
            job_cmd=job_cmd,
            partition='cpu-9m',
            log_dir='./logs',
            num_jobs_in_array=1
        )
        for method in cluster_methods:
            assignment_path = f'/home/space/diverse_priors/clustering/imagenet-subset-10k/{method}/num_clusters_{num_models}/cluster_labels.csv'
            job_cmd = f"""python clip_benchmark/sample_models.py  \
                        --num_samples {num_samples} \
                        --num_models {num_models} \
                        --sampling_strategies cluster one_cluster \
                        --cluster_assignment_path {assignment_path} \
                        --model_config_path {MODELS_CONFIG} \
                        --output_root {OUTPUT_ROOT}/models_{num_models}-samples_{num_samples}"""

            run_job(job_name=f"sampling",
                    job_cmd=job_cmd,
                    partition='cpu-9m',
                    log_dir='./logs',
                    num_jobs_in_array=1
                    )
