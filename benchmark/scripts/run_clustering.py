from slurm import run_job

SIM_MAT_ROOT = "/home/space/diverse_priors/model_similarities"
OUTPUT_ROOT = "/home/space/diverse_priors/clustering"

METHOD_KEYS = [
    'cka_kernel_rbf_unbiased_sigma_0.2',
    'cka_kernel_rbf_unbiased_sigma_0.4',
    'cka_kernel_rbf_unbiased_sigma_0.6',
    'cka_kernel_rbf_unbiased_sigma_0.8',
    'cka_kernel_linear_unbiased',
    'rsa_method_correlation_corr_method_pearson',
    'rsa_method_correlation_corr_method_spearman',
]

NUM_CLUSTERS = range(2, 12)

if __name__ == "__main__":
    for num_clusters in NUM_CLUSTERS:
        for method_key in METHOD_KEYS:
            job_cmd = f"""
            python ../clip_benchmark/cluster_models.py \
                --num_clusters {num_clusters} \
                --method_key {method_key} \
                --dataset imagenet-subset-10k \
                --assign_labels kmeans \
                --seed 0 \
                --sim_mat_root {SIM_MAT_ROOT} \
                --output_root {OUTPUT_ROOT}
            """
            run_job(
                job_name=f"clustering",
                job_cmd=job_cmd,
                partition='cpu-9m',
                log_dir=f'{OUTPUT_ROOT}/logs',
                num_jobs_in_array=1
            )