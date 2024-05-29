import os
from typing import Tuple, List, Optional

import numpy as np
from thingsvision.core.cka import get_cka
from thingsvision.core.rsa import compute_rdm, correlate_rdms
from tqdm import tqdm

from clip_benchmark.utils.utils import load_features, check_models


class BaseModelSimilarity:
    def __init__(self, feature_root: str, split: str = 'train', device: str = 'cuda') -> None:
        self.feature_root = feature_root
        self.split = split
        self.device = device
        self.model_ids = []

    def load_model_ids(self, model_ids: List[str]) -> None:
        assert os.path.exists(self.feature_root), "Feature root path non-existent"
        self.model_ids = check_models(self.feature_root, model_ids, self.split)
        self.model_ids_with_idx = [(i, model_id) for i, model_id in enumerate(self.model_ids)]

    def _prepare_sim_matrix(self) -> np.ndarray:
        return np.zeros((len(self.model_ids_with_idx), len(self.model_ids_with_idx)))

    def compute_similarity_matrix(self) -> Tuple[np.ndarray, List[str]]:
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_model_ids(self) -> List[str]:
        return self.model_ids


class CKAModelSimilarity(BaseModelSimilarity):
    def __init__(self, feature_root: str, split: str = 'train', device: str = 'cuda', kernel: str = 'linear',
                 backend: str = 'torch', unbiased: bool = True, sigma: Optional[float] = None) -> None:
        super().__init__(feature_root, split, device)
        self.kernel = kernel
        self.backend = backend
        self.unbiased = unbiased
        self.sigma = sigma

    def compute_similarity_matrix(self) -> Tuple[np.ndarray, List[str]]:
        sim_matrix = self._prepare_sim_matrix()
        for idx1, model1 in tqdm(self.model_ids_with_idx, desc="Computing CKA matrix"):
            features_i = load_features(self.feature_root, model1, self.split)
            for idx2, model2 in self.model_ids_with_idx:
                if idx1 >= idx2:
                    continue
                features_j = load_features(self.feature_root, model2, self.split)
                assert features_i.shape[0] == features_j.shape[0], \
                    f"Number of features should be equal for CKA computation. (model1: {model1}, model2: {model2})"

                m = features_i.shape[0]
                cka = get_cka(backend=self.backend, m=m, kernel=self.kernel, unbiased=self.unbiased, device=self.device,
                              sigma=self.sigma)
                rho = cka.compare(X=features_i, Y=features_j)
                sim_matrix[idx1, idx2] = rho
                sim_matrix[idx2, idx1] = rho

        return sim_matrix, self.model_ids

    def get_name(self) -> str:
        method_name = f"cka_kernel_{self.kernel}{'_unbiased' if self.unbiased else '_biased'}"
        if self.kernel == 'rbf':
            method_name += f"_sigma_{self.sigma}"
        return method_name


class RSAModelSimilarity(BaseModelSimilarity):
    def __init__(self, feature_root: str, split: str = 'train', device: str = 'cuda', rsa_method: str = 'correlation',
                 corr_method: str = 'spearman') -> None:
        super().__init__(feature_root, split, device)
        self.rsa_method = rsa_method
        self.corr_method = corr_method

    def compute_similarity_matrix(self) -> Tuple[np.ndarray, List[str]]:
        sim_matrix = self._prepare_sim_matrix()
        for idx1, model1 in tqdm(self.model_ids_with_idx, desc="Computing RSA matrix"):
            features_i = load_features(self.feature_root, model1, self.split).numpy()
            rdm_features_i = compute_rdm(features_i, method=self.rsa_method)
            for idx2, model2 in self.model_ids_with_idx:
                if idx1 >= idx2:
                    continue
                features_j = load_features(self.feature_root, model2, self.split).numpy()
                rdm_features_j = compute_rdm(features_j, method=self.rsa_method)
                assert features_i.shape[0] == features_j.shape[0], \
                    f"Number of features should be equal for RSA computation. (model1: {model1}, model2: {model2})"

                rho = correlate_rdms(rdm_features_i, rdm_features_j, correlation=self.corr_method)
                sim_matrix[idx1, idx2] = rho
                sim_matrix[idx2, idx1] = rho

        return sim_matrix, self.model_ids

    def get_name(self):
        if self.rsa_method == 'correlation':
            return f"rsa_method_{self.rsa_method}_corr_method_{self.corr_method}"
        else:
            return f"rsa_method_{self.rsa_method}"


def compute_sim_matrix(
        sim_method: str,
        feature_root: str,
        model_ids: List[str],
        split: str,
        kernel: str = 'linear',
        rsa_method: str = 'correlation',
        corr_method: str = 'spearman',
        backend: str = 'torch',
        unbiased: bool = True,
        device: str = 'cuda',
        sigma: Optional[float] = None
) -> Tuple[np.ndarray, List[str], str]:
    if sim_method == 'cka':
        model_similarity = CKAModelSimilarity(feature_root, split, device, kernel, backend, unbiased, sigma)
    elif sim_method == 'rsa':
        model_similarity = RSAModelSimilarity(feature_root, split, device, rsa_method, corr_method)
    else:
        raise ValueError(f"Unknown similarity method: {sim_method}")

    model_similarity.load_model_ids(model_ids)
    sim_mat, model_ids = model_similarity.compute_similarity_matrix()
    method_slug = model_similarity.get_name()
    return sim_mat, model_ids, method_slug
