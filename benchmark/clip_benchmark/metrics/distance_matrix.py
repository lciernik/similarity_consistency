import itertools
import os

import torch
from thingsvision.core.cka import CKA


def _load_features(feature_root, model_id, split='train'):
    """
    Load features from the given model and split
    Args:
        feature_root: Rootpath to the directory containing the features.
        model_id: Model id for which the features are to be loaded.
        split: Split for which the features are to be loaded. Default is 'train'.

    Returns:
        features: Features (torch.Tensor) for the given model and split.

    """
    features = torch.load(os.path.join(feature_root, model_id, f'features_{split}.pt'))
    return features


def _compute_cka_value(m, features_i, features_j, kernel='linear'):
    """
    Compute CKA value between two sets of features
    Args:
        m: Number of samples.
        features_i: Features from model i. Shape: (m, d_i)
        features_j: Features from model j. Shape: (m, d_j)
        kernel: Kernel to be used for CKA computation. Default is 'linear'.

    Returns:
        cka_value: CKA value between the two sets of features.

    """
    cka = CKA(m=m, kernel=kernel)
    return cka.compare(X=features_i, Y=features_j)


def compute_cka_matrix(feature_root, model_ids, split='train', kernel='linear'):
    """
    Compute CKA matrix for the given models
    Args:
        feature_root: Rootpath to the directory containing the features.
        model_ids: List of model ids for which CKA matrix is to be computed.
        split: Split for which the features are to be loaded. Default is 'train'.
        kernel: Kernel to be used for CKA computation. Default is 'linear'.
    Returns:
        cka_matrix: CKA matrix (torch.Tensor) for the given models. The matrix is symmetric.

    """

    assert os.path.exists(feature_root), "Feature root path non-existent"
    assert len(model_ids) > 1, "At least two models are required for CKA computation"

    model_ids_with_idx = [(i, model_id) for i, model_id in enumerate(model_ids)]

    cka_matrix = torch.zeros(len(model_ids),
                             len(model_ids))  # Initialize CKA matrix. What is the CKA of a model with itself?

    for ((idx1, model1), (idx2, model2)) in itertools.combinations(model_ids_with_idx, 2):
        features_i = _load_features(feature_root, model1, split)
        features_j = _load_features(feature_root, model2, split)

        assert features_i.shape[0] == features_j.shape[
            0], f"Number of features should be equal for CKA computation. (model1: {model1}, model2: {model2})"

        m = features_i.shape[0]
        cka_value = _compute_cka_value(m, features_i, features_j, kernel)
        cka_matrix[idx1, idx2] = cka_value
        cka_matrix[idx2, idx1] = cka_value

    return cka_matrix
