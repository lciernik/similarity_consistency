import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import ot
from scipy.spatial.distance import cdist
from thingsvision.core.cka import get_cka
from thingsvision.core.rsa import compute_rdm, correlate_rdms
from tqdm import tqdm

from sim_consistency.utils.utils import load_features, check_models


class BaseModelSimilarity:
    def __init__(self, feature_root: str, subset_root: Optional[str], split: str = 'train', device: str = 'cuda',
                 max_workers: int = 4) -> None:
        self.feature_root = feature_root
        self.split = split
        self.device = device
        self.model_ids = []
        self.max_workers = max_workers
        self.subset_indices = self._load_subset_indices(subset_root)
        self.name = 'Base'

    def _load_subset_indices(self, subset_root) -> Optional[List[int]]:
        subset_path = os.path.join(subset_root, f'subset_indices_{self.split}.json')
        if not os.path.exists(subset_path):
            warnings.warn(
                f"Subset indices not found at {subset_path}. Continuing with full datasets."
            )
            return None
        with open(subset_path, 'r') as f:
            subset_indices = json.load(f)
        return subset_indices

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
    def __init__(self, feature_root: str, subset_root: Optional[str], split: str = 'train', device: str = 'cuda',
                 kernel: str = 'linear', backend: str = 'torch', unbiased: bool = True, sigma: Optional[float] = None,
                 max_workers: int = 4) -> None:
        super().__init__(feature_root=feature_root, subset_root=subset_root, split=split, device=device,
                         max_workers=max_workers)
        self.kernel = kernel
        self.backend = backend
        self.unbiased = unbiased
        self.sigma = sigma

    # def _load_feature(self, model_id:str) -> np.ndarray:
    #     features = load_features(self.feature_root, model_id, self.split, self.subset_indices)
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
            features_i = load_features(self.feature_root, model1, self.split, self.subset_indices)
            for idx2, model2 in self.model_ids_with_idx:
                if idx1 >= idx2:
                    continue
                features_j = load_features(self.feature_root, model2, self.split, self.subset_indices)
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
    def __init__(self, feature_root: str, subset_root: Optional[str], split: str = 'train', device: str = 'cuda',
                 rsa_method: str = 'correlation', corr_method: str = 'spearman', max_workers: int = 4) -> None:
        super().__init__(feature_root=feature_root, subset_root=subset_root, split=split, device=device,
                         max_workers=max_workers)
        self.rsa_method = rsa_method
        self.corr_method = corr_method
        self.name = 'RSA'

    def _load_feature(self, model_id: str) -> np.ndarray:
        features = load_features(self.feature_root, model_id, self.split, self.subset_indices).numpy()
        rdm_features = compute_rdm(features, method=self.rsa_method)
        return rdm_features

    def _compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        return correlate_rdms(feat1, feat2, correlation=self.corr_method)

    def get_name(self):
        if self.rsa_method == 'correlation':
            return f"rsa_method_{self.rsa_method}_corr_method_{self.corr_method}"
        else:
            return f"rsa_method_{self.rsa_method}"


class GWModelSimilarity(BaseModelSimilarity):
    def __init__(
            self,
            feature_root: str,
            subset_root: Optional[str],
            split: str = 'train',
            device: str = 'cuda',
            cost_fun: str = 'euclidean',
            gromov_type: str = 'fixed_coupling',
            loss_fun: str = 'square_loss',
            max_workers: int = 4,
            store_coupling: bool = False,
            output_root: Optional[str] = None,
    ) -> None:
        super().__init__(feature_root=feature_root, subset_root=subset_root, split=split, device=device,
                         max_workers=max_workers)

        self.output_root = None

        if store_coupling:
            assert output_root is not None, "Output root should be provided for storing coupling matrices"
            self.output_root = Path(output_root)
            assert self.output_root.exists(), "Output root path does not exist"

        self.store_coupling = store_coupling

        if cost_fun not in ['euclidean', 'cosine']:
            raise ValueError(f"Unknown cost function: {cost_fun}")
        else:
            self.cost_fun = cost_fun

        if loss_fun not in ['square_loss', 'kl_loss']:
            raise ValueError(f"Unknown loss function: {loss_fun}")
        else:
            self.loss_fun = loss_fun
        if gromov_type not in ['fixed_coupling', 'full_gromov', 'sampled_gromov', 'entropic_gromov']:
            raise ValueError(f"Unknown gromov type: {gromov_type}")
        else:
            self.gromov_type = gromov_type

    def _prepare_sim_matrix(self) -> np.ndarray:
        return np.zeros((len(self.model_ids_with_idx), len(self.model_ids_with_idx)))

    def _load_feature(self, model_id: str) -> np.ndarray:
        features = load_features(self.feature_root, model_id, self.split, self.subset_indices).numpy()
        C_mat = cdist(features.numpy(), features.numpy(), metric=self.cost_fun)
        C_mat /= C_mat.max()
        return C_mat

    def get_name(self):
        return f"gw_sim_{self.gromov_type}_cost_{self.cost_fun}_loss_fun_{self.loss_fun}"

    def store_coupling_matrix(self, model1: str, model2: str, coupling_matrix: np.ndarray) -> None:
        if self.store_coupling:
            output_path = self.output_root / f"{model1}_{model2}_coupling.npy"
            np.save(output_path, coupling_matrix)

    def _comput_gromov_distance(self, C1: np.ndarray, C2: np.ndarray) -> (float, np.ndarray):
        if self.gromov_type == "fixed_coupling":
            T = np.eye(C1.shape[0])
            if self.loss_fun == "square_loss":
                gw_loss = np.mean((C1 - C2) ** 2)
            else:
                raise NotImplementedError("Currently do not support KL Loss for fixed coupling")
        elif self.gromov_type == "full_gromov":
            gw_loss, log_gw = ot.gromov.gromov_wasserstein2(C1, C2, loss_fun=self.loss_fun, log=True)
            T = log_gw['T']
        elif self.gromov_type == "sampled_gromov":
            p = ot.utils.unif(C1.shape[0], type_as=C1)
            q = ot.utils.unif(C2.shape[0], type_as=C2)
            T, log_gw = ot.gromov.sampled_gromov_wasserstein(C1, C2, p, q, loss_fun=self.loss_fun, log=True)
            gw_loss = log_gw["gw_dist_estimated"]
            # We could also check stability with log["gw_dist_std]
        elif self.gromov_type == "entropic_gromov":
            gw_loss, log_gw = ot.gromov.entropic_gromov_wasserstein2(C1, C2, loss_fun=self.loss_fun, log=True)
            T = log_gw['T']
        else:
            raise NotImplementedError(f"Unknown gromov type: {self.gromov_type}")
        # We need to take the square root to get the distance out of the gw_loss computed by OT
        return 0.5 * gw_loss**0.5, T

    def compute_similarity_matrix(self) -> np.ndarray:
        dist_matrix = self._prepare_sim_matrix()
        for idx1, model1 in tqdm(self.model_ids_with_idx, desc=f"Computing CKA matrix"):
            C_i = load_features(self.feature_root, model1, self.split, self.subset_indices)
            for idx2, model2 in self.model_ids_with_idx:
                if idx1 >= idx2:
                    continue
                C_j = load_features(self.feature_root, model2, self.split, self.subset_indices)

                assert C_i.shape[0] == C_j.shape[0], \
                    (f"Number of samples should be equal for both models. (model1: {model1}, model2: {model2},"
                     f"feature_root: {self.feature_root})")

                gw_dist, T = self._comput_gromov_distance(C_i, C_j)
                self.store_coupling_matrix(model1, model2, T)

                dist_matrix[idx1, idx2] = gw_dist
                dist_matrix[idx2, idx1] = gw_dist
        return dist_matrix

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
        gromov_cost_fun: str = 'euclidean',
        gromov_type: str = 'fixed_coupling',
        gromov_loss_fun: str = 'square_loss',
        gromov_store_coupling: bool = False,
        output_root: Optional[str] = None,
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
    elif sim_method =='gromov':
        model_similarity = GWModelSimilarity(
            feature_root=feature_root,
            subset_root=subset_root,
            split=split,
            device=device,
            cost_fun=gromov_cost_fun,
            gromov_type=gromov_type,
            loss_fun=gromov_loss_fun,
            max_workers=max_workers,
            store_coupling=gromov_store_coupling,
            output_root=output_root
        )


    else:
        raise ValueError(f"Unknown similarity method: {sim_method}")

    model_similarity.load_model_ids(model_ids)
    model_ids = model_similarity.get_model_ids()
    sim_mat = model_similarity.compute_similarity_matrix()
    method_slug = model_similarity.get_name()
    return sim_mat, model_ids, method_slug
