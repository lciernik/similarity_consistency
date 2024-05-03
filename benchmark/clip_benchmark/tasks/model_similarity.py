import os

import numpy as np
from thingsvision.core.cka import get_cka
from thingsvision.core.rsa import compute_rdm, correlate_rdms
from tqdm import tqdm

from clip_benchmark.utils.utils import load_features, check_models


def compute_cka_matrix(model_ids_with_idx, feature_root, split='train', kernel='linear',
                       backend='torch', unbiased=True, device='cuda', sigma=None):
    cka_matrix = np.zeros((len(model_ids_with_idx), len(model_ids_with_idx)))
    for idx1, model1 in tqdm(model_ids_with_idx, desc="Computing CKA matrix"):
        features_i = load_features(feature_root, model1, split)
        for idx2, model2 in model_ids_with_idx:
            if idx1 >= idx2:
                continue
            features_j = load_features(feature_root, model2, split)
            assert features_i.shape[0] == features_j.shape[
                0], f"Number of features should be equal for CKA computation. (model1: {model1}, model2: {model2})"

            m = features_i.shape[0]
            cka = get_cka(backend=backend, m=m, kernel=kernel, unbiased=unbiased, device=device, sigma=sigma)
            rho = cka.compare(X=features_i, Y=features_j)
            cka_matrix[idx1, idx2] = rho
            cka_matrix[idx2, idx1] = rho

    return cka_matrix


def compute_rsa_matrix(model_ids_with_idx, feature_root, split='train', rsa_method='correlation',
                       corr_method='spearman'):
    rsa_matrix = np.zeros((len(model_ids_with_idx), len(model_ids_with_idx)))
    for idx1, model1 in tqdm(model_ids_with_idx, desc="Computing RSA matrix"):
        features_i = load_features(feature_root, model1, split).numpy()
        rdm_features_i = compute_rdm(features_i, method=rsa_method)
        for idx2, model2 in model_ids_with_idx:
            if idx1 >= idx2:
                continue
            features_j = load_features(feature_root, model2, split).numpy()
            rdm_features_j = compute_rdm(features_j, method=rsa_method)
            assert features_i.shape[0] == features_j.shape[
                0], f"Number of features should be equal for RSA computation. (model1: {model1}, model2: {model2})"

            rho = correlate_rdms(rdm_features_i, rdm_features_j, correlation=corr_method)
            rsa_matrix[idx1, idx2] = rho
            rsa_matrix[idx2, idx1] = rho

    return rsa_matrix


def compute_sim_matrix(sim_method, feature_root, model_ids, split, kernel='linear', rsa_method='correlation',
                       corr_method='spearman', backend='torch', unbiased=True, device='cuda', sigma=None):
    assert os.path.exists(feature_root), "Feature root path non-existent"

    model_ids = check_models(feature_root, model_ids, split)

    model_ids_with_idx = [(i, model_id) for i, model_id in enumerate(model_ids)]

    if sim_method == 'cka':
        sim_matrix = compute_cka_matrix(model_ids_with_idx, feature_root,
                                         split=split, kernel=kernel, backend=backend,
                                         unbiased=unbiased, device=device, sigma=sigma)
    elif sim_method == 'rsa':
        sim_matrix = compute_rsa_matrix(model_ids_with_idx, feature_root, split=split, rsa_method=rsa_method,
                                         corr_method=corr_method)
    else:
        raise ValueError(f"Unknown method to compute distance matrix between the models.")

    return sim_matrix, model_ids
