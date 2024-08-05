import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Optional

import numpy as np
from thingsvision.core.cka import get_cka
from thingsvision.core.rsa import compute_rdm, correlate_rdms
from tqdm import tqdm

from clip_benchmark.utils.utils import load_features, check_models


class BaseModelSimilarity:
    def __init__(self, feature_root: str, subset_root: Optional[str], split: str = 'train', device: str = 'cuda',
                 max_workers: int = 4) -> None:
        self.feature_root = feature_root
        self.subset_root = subset_root
        self.split = split
        self.device = device
        self.model_ids = []
        self.max_workers = max_workers
        self.name = 'Base'

    def load_model_ids(self, model_ids: List[str]) -> None:
        assert os.path.exists(self.feature_root), "Feature root path non-existent"
        self.model_ids = check_models(self.feature_root, model_ids, self.split)
        self.model_ids_with_idx = [(i, model_id) for i, model_id in enumerate(self.model_ids)]

    def _prepare_sim_matrix(self) -> np.ndarray:
        return np.ones((len(self.model_ids_with_idx), len(self.model_ids_with_idx)))

    def _load_feature(self, model_id: str) -> np.ndarray:
        raise NotImplementedError()

    def _compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        raise NotImplementedError()

    def compute_pairwise_similarity(self, features_1: np.ndarray, model2: str) -> float:
        features_2 = self._load_feature(model_id=model2)

        assert features_1.shape[0] == features_2.shape[0], (f"Number of features should be equal for "
                                                            f"similarity computation.")

        rho = self._compute_similarity(features_1, features_2)
        return rho

    def compute_similarity_matrix(self) -> np.ndarray:
        sim_matrix = self._prepare_sim_matrix()
        max_workers = self.max_workers
        for idx1, model1 in tqdm(self.model_ids_with_idx, desc=f"Computing {self.name} matrix"):
            features_1 = self._load_feature(model_id=model1)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for idx2, model2 in self.model_ids_with_idx:
                    if idx1 < idx2:
                        future = executor.submit(self.compute_pairwise_similarity, features_1, model2)
                        futures[future] = (idx1, idx2)

                for future in tqdm(as_completed(futures), total=len(futures), desc="Pairwise similarity computation"):
                    cidx1, cidx2 = futures[future]
                    rho = future.result()
                    sim_matrix[cidx1, cidx2] = rho
        upper_tri = np.triu(sim_matrix)
        sim_matrix = upper_tri + upper_tri.T - np.diag(np.diag(sim_matrix))
        return sim_matrix

    def get_model_ids(self) -> List[str]:
        return self.model_ids


class CKAModelSimilarity(BaseModelSimilarity):
    def __init__(self, feature_root: str, subset_root: Optional[str], split: str = 'train', device: str = 'cuda', kernel: str = 'linear',
                 backend: str = 'torch', unbiased: bool = True, sigma: Optional[float] = None, max_workers: int = 4) -> None:
        super().__init__(feature_root=feature_root, subset_root=subset_root, split=split, device=device, max_workers=max_workers)
        self.kernel = kernel
        self.backend = backend
        self.unbiased = unbiased
        self.sigma = sigma

    # def _load_feature(self, model_id:str) -> np.ndarray:
    #     features = load_features(self.feature_root, model_id, self.split, self.subset_root)
    #     return features

    # def _compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
    #     m = feat1.shape[0]
    #     cka = get_cka(backend=self.backend, m=m, kernel=self.kernel, unbiased=self.unbiased, device=self.device,
    #                     sigma=self.sigma)
    #     rho = cka.compare(X=feat1, Y=feat2)
    #     return rho

    def compute_similarity_matrix(self) -> np.ndarray:
        sim_matrix = self._prepare_sim_matrix()
        for idx1, model1 in tqdm(self.model_ids_with_idx, desc=f"Computing CKA matrix"):
            features_i = load_features(self.feature_root, model1, self.split, self.subset_root)
            for idx2, model2 in self.model_ids_with_idx:
                if idx1 >= idx2:
                    continue
                features_j = load_features(self.feature_root, model2, self.split, self.subset_root)
                assert features_i.shape[0] == features_j.shape[0], \
                    f"Number of features should be equal for CKA computation. (model1: {model1}, model2: {model2})"

                m = features_i.shape[0]
                cka = get_cka(backend=self.backend, m=m, kernel=self.kernel, unbiased=self.unbiased, device=self.device,
                              sigma=self.sigma)
                rho = cka.compare(X=features_i, Y=features_j)
                sim_matrix[idx1, idx2] = rho
                sim_matrix[idx2, idx1] = rho

        return sim_matrix

    def get_name(self) -> str:
        method_name = f"cka_kernel_{self.kernel}{'_unbiased' if self.unbiased else '_biased'}"
        if self.kernel == 'rbf':
            method_name += f"_sigma_{self.sigma}"
        return method_name


class RSAModelSimilarity(BaseModelSimilarity):
    def __init__(self, feature_root: str, subset_root: Optional[str], split: str = 'train', device: str = 'cuda', rsa_method: str = 'correlation',
                 corr_method: str = 'spearman', max_workers: int = 4) -> None:
        super().__init__(feature_root=feature_root, subset_root=subset_root, split=split, device=device, max_workers=max_workers)
        self.rsa_method = rsa_method
        self.corr_method = corr_method
        self.name = 'RSA'

    def _load_feature(self, model_id: str) -> np.ndarray:
        features = load_features(self.feature_root, model_id, self.split, self.subset_root).numpy()
        rdm_features = compute_rdm(features, method=self.rsa_method)
        return rdm_features

    def _compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        return correlate_rdms(feat1, feat2, correlation=self.corr_method)

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
        subset_root: Optional[str] = None,
        kernel: str = 'linear',
        rsa_method: str = 'correlation',
        corr_method: str = 'spearman',
        backend: str = 'torch',
        unbiased: bool = True,
        device: str = 'cuda',
        sigma: Optional[float] = None,
        max_workers: int = 4,
) -> Tuple[np.ndarray, List[str], str]:
    if sim_method == 'cka':
        model_similarity = CKAModelSimilarity(
            feature_root=feature_root,
            subset_root=subset_root, 
            split=split, 
            device=device, 
            kernel=kernel, 
            backend=backend, 
            unbiased=unbiased, 
            sigma=sigma,
            max_workers=max_workers
            )
    elif sim_method == 'rsa':
        model_similarity = RSAModelSimilarity(
            feature_root=feature_root, 
            subset_root=subset_root, 
            split=split, 
            device=device, 
            rsa_method=rsa_method, 
            corr_method=corr_method, 
            max_workers=max_workers
            )
    else:
        raise ValueError(f"Unknown similarity method: {sim_method}")

    model_similarity.load_model_ids(model_ids)
    model_ids = model_similarity.get_model_ids()
    sim_mat = model_similarity.compute_similarity_matrix()
    method_slug = model_similarity.get_name()
    return sim_mat, model_ids, method_slug
