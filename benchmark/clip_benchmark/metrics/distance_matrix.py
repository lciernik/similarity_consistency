import os

import numpy as np
import torch
from thingsvision.core.cka import get_cka
from thingsvision.core.rsa import compute_rdm, correlate_rdms
from tqdm import tqdm


def _load_features(feature_root, model_id, split='train'):
    features = torch.load(os.path.join(feature_root, model_id, f'features_{split}.pt'))
    return features


def _check_models(feature_root, model_ids, split):
    prev_model_ids = model_ids

    model_ids = sorted(
        [mid for mid in model_ids if os.path.exists(os.path.join(feature_root, mid, f'features_{split}.pt'))])

    if len(set(prev_model_ids)) != len(set(model_ids)):
        print(f"Features do not exist for the following models: {set(prev_model_ids) - set(model_ids)}")
        print(f"Removing the above models from the list of models for distance computation.")

    # Check if enough remaining models to compute distance matrix
    assert len(model_ids) > 1, f"At least two models are required for distance computation"

    return model_ids


def compute_cka_matrix(cka_matrix, model_ids_with_idx, feature_root, split='train', kernel='linear',
                       backend='torch', unbiased=True, device='cuda', sigma=None):
    for idx1, model1 in tqdm(model_ids_with_idx, desc="Computing CKA matrix"):
        features_i = _load_features(feature_root, model1, split).numpy()
        for idx2, model2 in model_ids_with_idx:
            if idx1 >= idx2:
                continue
            features_j = _load_features(feature_root, model2, split).numpy()
            assert features_i.shape[0] == features_j.shape[
                0], f"Number of features should be equal for CKA computation. (model1: {model1}, model2: {model2})"

            m = features_i.shape[0]
            cka = get_cka(backend=backend, m=m, kernel=kernel, unbiased=unbiased, device=device, sigma=sigma)
            rho = cka.compare(X=features_i, Y=features_j)
            cka_matrix[idx1, idx2] = rho
            cka_matrix[idx2, idx1] = rho

    return cka_matrix


def compute_rsa_matrix(rsa_matrix, model_ids_with_idx, feature_root, split='train', rsa_method='correlation',
                       corr_method='spearman'):
    for idx1, model1 in tqdm(model_ids_with_idx, desc="Computing RSA matrix"):
        features_i = _load_features(feature_root, model1, split).numpy()
        rdm_features_i = compute_rdm(features_i, method=rsa_method)
        for idx2, model2 in model_ids_with_idx:
            if idx1 >= idx2:
                continue
            features_j = _load_features(feature_root, model2, split).numpy()
            rdm_features_j = compute_rdm(features_j, method=rsa_method)
            assert features_i.shape[0] == features_j.shape[
                0], f"Number of features should be equal for RSA computation. (model1: {model1}, model2: {model2})"

            rho = correlate_rdms(rdm_features_i, rdm_features_j, correlation=corr_method)
            rsa_matrix[idx1, idx2] = rho
            rsa_matrix[idx2, idx1] = rho

    return rsa_matrix


def compute_dist_matrix(sim_method, feature_root, model_ids, split, kernel='linear', rsa_method='correlation',
                        corr_method='spearman'):
    assert os.path.exists(feature_root), "Feature root path non-existent"

    model_ids = _check_models(feature_root, model_ids, split)

    model_ids_with_idx = [(i, model_id) for i, model_id in enumerate(model_ids)]

    dist_matrix = np.zeros((len(model_ids_with_idx), len(model_ids_with_idx)))  # Initialize CKA matrix.

    if sim_method == 'cka':
        compute_cka_matrix(dist_matrix, model_ids_with_idx, feature_root, split=split, kernel=kernel)
    elif sim_method == 'rsa':
        compute_rsa_matrix(dist_matrix, model_ids_with_idx, feature_root, split=split, rsa_method=rsa_method,
                           corr_method=corr_method)
    else:
        raise ValueError(f"Unknown method to compute distance matrix between the models.")

    return dist_matrix, model_ids
