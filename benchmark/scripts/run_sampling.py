import os

from slurm import run_job

MODELS_CONFIG = "./scripts/models_config.json"
BASE_PROJECT_PATH = "/home/space/diverse_priors"
OUTPUT_ROOT = os.path.join(BASE_PROJECT_PATH, 'sampling')
CLUSTERING_ROOT = os.path.join(BASE_PROJECT_PATH, 'clustering')
SELECTION_DATASET = 'wds_imagenet1k'

METHOD_KEYS = [
    'cka_kernel_rbf_unbiased_sigma_0.2',
    'cka_kernel_rbf_unbiased_sigma_0.4',
    'cka_kernel_rbf_unbiased_sigma_0.6',
    'cka_kernel_rbf_unbiased_sigma_0.8',
    'cka_kernel_linear_unbiased',
    'rsa_method_correlation_corr_method_pearson',
    'rsa_method_correlation_corr_method_spearman',
]

NUM_CLUSTERS = range(3, 8)

LBL_ASSIGN_METHODS = ['kmeans', 'discretize', 'cluster_qr']

if __name__ == "__main__":
    # For samplers with randomness, we want to get multiple model sets to get a better estimate of the performance.
    num_samples = 10
    # Iterate over different number of models we want to use in the combined setting
    for num_models in range(3, 6):
        ### Run model selection with Top-K and Random sampling
        job_cmd = f"""python clip_benchmark/sample_models.py  \
            --num_models {num_models} \
            --num_samples {num_samples} \
            --sampling_strategies top-k random \
            --model_config_path {MODELS_CONFIG} \
            --selection_dataset {SELECTION_DATASET} \
            --output_root {OUTPUT_ROOT}/models_{num_models}-samples_{num_samples}"""

        run_job(
            job_name=f"sampling",
            job_cmd=job_cmd,
            partition='cpu-2h',
            log_dir='./logs',
            num_cpus=1,
            num_jobs_in_array=1
        )

        ### Run model selection with Top-K and Random sampling
        # Iterate over the different similarity measures
        for method in METHOD_KEYS:
            # Iterate over the different number of clusters
            for num_clusters in NUM_CLUSTERS:
                for lbl_assign_method in LBL_ASSIGN_METHODS:
                    assignment_path = os.path.join(CLUSTERING_ROOT,
                                                   'imagenet-subset-10k',
                                                   method,
                                                   f"num_clusters_{num_clusters}",
                                                   lbl_assign_method,
                                                   'cluster_labels.csv')
                    cluster_slug = f"""{method}_num_clusters_{num_clusters}_{lbl_assign_method}"""
                    job_cmd = f"""python clip_benchmark/sample_models.py  \
                                            --num_models {num_models} \
                                            --num_samples {num_samples} \
                                            --selection_dataset {SELECTION_DATASET} \
                                            --sampling_strategies cluster_best cluster_random one_cluster \
                                            --cluster_assignment_path {assignment_path} \
                                            --cluster_slug {cluster_slug} \
                                            --model_config_path {MODELS_CONFIG} \
                                            --output_root {OUTPUT_ROOT}/models_{num_models}-samples_{num_samples}"""

                    run_job(job_name=f"sampling",
                            job_cmd=job_cmd,
                            partition='cpu-2h',
                            log_dir='./logs',
                            num_jobs_in_array=1,
                            num_cpus=1
                            )
