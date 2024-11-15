import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class BaseFeatureCombiner:
    def __init__(self, reference_combiner=None):
        self.features = None
        self.reference_combiner = reference_combiner

    def __getitem__(self, i):
        return self.features[i]

    def __call__(self, i):
        return self.features[i]

    def set_features(self, list_features, normalize=True):
        self.features = list_features
        if normalize:
            self.features = F.normalize(self.features, dim=-1)


class ConcatFeatureCombiner(BaseFeatureCombiner):
    def set_features(self, list_features, normalize=True):
        self.features = torch.concat(list_features, dim=1)
        if normalize:
            self.features = F.normalize(self.features, dim=-1)


class PCAConcatFeatureCombiner(BaseFeatureCombiner):
    def __init__(self, pct_var=0.99, reference_combiner=None):
        super().__init__()
        if reference_combiner is None:
            self.pca = PCA()
            self.scalar = StandardScaler()
            self.pct_var = pct_var
            self.n_components = None
            self.scale_fn = self.scalar.fit_transform
            self.pca_fn = self.pca.fit_transform

        else:
            assert isinstance(reference_combiner,
                              PCAConcatFeatureCombiner), "Reference combiner should be a PCAConcatFeatureCombiner"
            assert reference_combiner.features is not None, "Reference combiner should have features set"

            self.pca = reference_combiner.pca
            self.scalar = reference_combiner.scalar
            self.pct_var = reference_combiner.pct_var
            self.n_components = reference_combiner.n_components
            self.scale_fn = self.scalar.transform
            self.pca_fn = self.pca.transform

    def set_features(self, list_features, normalize=True):
        features = torch.concat(list_features, dim=1)
        scaled_features = self.scale_fn(features)
        pca_features = self.pca_fn(scaled_features)
        if self.n_components is None:
            self.n_components = np.argmax(np.cumsum(self.pca.explained_variance_ratio_) > self.pct_var) + 1
        self.features = torch.Tensor(pca_features[:, :self.n_components])
        if normalize:
            self.features = F.normalize(self.features, dim=-1)


def get_feature_combiner_cls(combiner_name):
    if combiner_name == "concat":
        return ConcatFeatureCombiner
    elif combiner_name == "concat_pca":
        return PCAConcatFeatureCombiner
    else:
        raise ValueError(f"Unknown feature combiner name: {combiner_name}")
