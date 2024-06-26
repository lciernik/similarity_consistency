import os
from typing import List

import pandas as pd

from helper import load_models
from slurm import run_job

MODELS_CONFIG = "./models_config.json"
BASE_PROJECT_PATH = "/home/space/diverse_priors"
OUTPUT_ROOT = os.path.join(BASE_PROJECT_PATH, 'sampling')
CLUSTERING_ROOT = os.path.join(BASE_PROJECT_PATH, 'clustering')
SELECTION_DATASET = 'wds_imagenet1k'
DATASET = "imagenet-subset-10k"

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
NUM_MODELS = range(3, 8)

# LBL_ASSIGN_METHODS = ['kmeans', 'discretize', 'cluster_qr']
LBL_ASSIGN_METHODS = ['cluster_qr']


def run_curr_job(job_cmd: str) -> None:
    run_job(
        job_name=f"sampling",
        job_cmd=job_cmd,
        partition='cpu-2h',
        log_dir=f'{OUTPUT_ROOT}/logs',
        num_cpus=1,
        num_jobs_in_array=1,
    )


def get_base_cmd(num_models: int, num_samples: int, model_keys: List[str]) -> str:
    return f"""python ../clip_benchmark/sample_models.py \
            --num_models {num_models} \
            --num_samples {num_samples} \
            --model_key {model_keys} \
            --model_config_path {MODELS_CONFIG} \
            --selection_dataset {SELECTION_DATASET} \
            --output_root {OUTPUT_ROOT}/models_{num_models}-samples_{num_samples}"""


if __name__ == "__main__":
    # For samplers with randomness, we want to get multiple model sets to get a better estimate of the performance.
    num_samples = 10

    models, n_models = load_models(MODELS_CONFIG)
    models.pop('SegmentAnything_vit_b')
    model_keys = ' '.join(models.keys())

    # Iterate over different number of models we want to use in the combined setting
    for num_models in NUM_MODELS:
        ### Run model selection with Top-K and Random sampling
        base_job_cmd = get_base_cmd(num_models, num_samples, model_keys)
        job_cmd = base_job_cmd + " --sampling_strategies top-k random"

        run_curr_job(job_cmd)

        # Iterate over the different similarity measures
        for method in METHOD_KEYS:
            # Iterate over the different number of clusters
            for num_clusters in NUM_CLUSTERS:

                # NOTE: this is not nicely done, maybe in future we want k!=num_clusters 
                if num_clusters != num_models:
                    continue

                for lbl_assign_method in LBL_ASSIGN_METHODS:

                    # run 'cluster_best'
                    assignment_path = os.path.join(CLUSTERING_ROOT,
                                                   DATASET,
                                                   method,
                                                   f"num_clusters_{num_clusters}",
                                                   lbl_assign_method,
                                                   'cluster_labels.csv')

                    cluster_slug = f"""{method}-num_clusters_{num_clusters}-{lbl_assign_method}"""
                    job_cmd = base_job_cmd + f""" --sampling_strategies cluster_best \
                    --cluster_assignment_path {assignment_path} \
                    --cluster_slug {cluster_slug}"""

                    run_curr_job(job_cmd)

                    # # run 'cluster_best' and 'cluster_random'
                    # assignment_path = os.path.join(CLUSTERING_ROOT,
                    #                                DATASET,
                    #                                method,
                    #                                f"num_clusters_{num_clusters}",
                    #                                lbl_assign_method,
                    #                                'cluster_labels.csv')

                    # cluster_slug = f"""{method}-num_clusters_{num_clusters}-{lbl_assign_method}"""
                    # job_cmd = base_job_cmd + f""" --sampling_strategies cluster_best cluster_random \
                    # --cluster_assignment_path {assignment_path} \
                    # --cluster_slug {cluster_slug}"""

                    # run_curr_job(job_cmd)

                    # # run 'one_cluster' for each cluster in cluster assignment
                    # cluster_assignment_table = pd.read_csv(assignment_path)
                    # for cluster_idx in cluster_assignment_table.cluster.unique():
                    #     if len(cluster_assignment_table[cluster_assignment_table.cluster == cluster_idx]) >= num_clusters:
                    #         curr_cluster_slug = f"""{method}-num_clusters_{num_clusters}-{lbl_assign_method}-cluster_{cluster_idx}"""
                    #         job_cmd = base_job_cmd + f""" --sampling_strategies one_cluster \
                    #         --cluster_assignment_path {assignment_path} \
                    #         --cluster_index {cluster_idx} \
                    #         --cluster_slug {curr_cluster_slug}"""

                    #         run_curr_job(job_cmd)
