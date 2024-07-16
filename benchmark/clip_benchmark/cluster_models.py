import argparse
import os
from typing import Union, List

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import SpectralClustering


def check_paths(args: argparse.Namespace):
    sim_mat_path = os.path.join(args.sim_mat_root, args.dataset, args.method_key, 'similarity_matrix.pt')
    model_ids_path = os.path.join(args.sim_mat_root, args.dataset, args.method_key, 'model_ids.txt')
    if not os.path.exists(sim_mat_path):
        raise FileNotFoundError(f"Similarity matrix not found at {sim_mat_path}")
    if not os.path.exists(model_ids_path):
        raise FileNotFoundError(f"Model ids not found at {model_ids_path}")
    return sim_mat_path, model_ids_path


def load_similarity_matrix(sim_mat_path: Union[str, os.PathLike], model_ids_path: Union[str, os.PathLike]):
    sim_mat = torch.load(sim_mat_path)
    with open(model_ids_path, 'r') as f:
        model_ids = f.read().splitlines()
    return pd.DataFrame(sim_mat, index=model_ids, columns=model_ids)


def process_similarity_matrix(sim_mat: pd.DataFrame, allowed_models: List[str]) -> pd.DataFrame:
    # filter only desired models
    sim_mat = sim_mat.loc[allowed_models, allowed_models].copy()

    sim_mat = sim_mat.abs()
    if not np.all(np.diag(sim_mat.values) == 1):
        np.fill_diagonal(sim_mat.values, 1)
    return sim_mat


def main(args: argparse.Namespace):
    sim_mat_path, model_ids_path = check_paths(args)
    sim_mat = load_similarity_matrix(sim_mat_path, model_ids_path)
    sim_mat = process_similarity_matrix(sim_mat, args.model_key)

    clustering = SpectralClustering(n_clusters=args.num_clusters,
                                    affinity='precomputed',
                                    assign_labels=args.assign_labels,
                                    random_state=args.seed)

    labels = clustering.fit_predict(sim_mat.values, y=None)
    labels = pd.DataFrame({'model_id': sim_mat.index.to_numpy(), 'cluster': labels}, index=sim_mat.index)

    output_path = os.path.join(
        args.output_root,
        args.dataset,
        args.method_key,
        f"num_clusters_{args.num_clusters}",
        args.assign_labels
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    labels.to_csv(os.path.join(output_path, 'cluster_labels.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clusters', type=int, help="Number of clusters to create.")
    parser.add_argument('--method_key', type=str,
                        choices=[
                            'cka_kernel_rbf_unbiased_sigma_0.2',
                            'cka_kernel_rbf_unbiased_sigma_0.4',
                            'cka_kernel_rbf_unbiased_sigma_0.6',
                            'cka_kernel_rbf_unbiased_sigma_0.8',
                            'cka_kernel_linear_unbiased',
                            'rsa_method_correlation_corr_method_pearson',
                            'rsa_method_correlation_corr_method_spearman',
                        ],
                        default='rsa_correlation_spearman')

    parser.add_argument('--dataset', type=str, default='imagenet-subset-10k',
                        help="Dataset to use for clustering.")
    parser.add_argument('--model_key', type=str, nargs='+', help="Define the models we want to consider during clustering.")
    parser.add_argument('--assign_labels', type=str, default='kmeans',
                        choices=['kmeans', 'discretize', 'cluster_qr'],
                        help="Method used to assign labels during SpectralClustering.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument('--sim_mat_root', type=str, default="/home/space/diverse_priors/model_similarities")
    parser.add_argument('--output_root', type=str, default="/home/space/diverse_priors/clustering")

    args = parser.parse_args()
    main(args)
