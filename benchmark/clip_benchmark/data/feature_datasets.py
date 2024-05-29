from torch.utils.data import Dataset

__all__ = ['FeatureDataset', 'CombinedFeaturesDataset']


class FeatureDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.targets[i]


class CombinedFeaturesDataset(Dataset):
    def __init__(self, list_features, targets, feature_combiner, normalize=True):
        if not isinstance(list_features, list):
            self.list_features = [list_features]
        else:
            self.list_features = list_features
        self.targets = targets
        self.nr_comb_feats = len(list_features)
        self.feature_combiner = feature_combiner
        self.feature_combiner.set_features(self.list_features, normalize)

    def __len__(self):
        return len(self.list_features[0])

    def __getitem__(self, i):
        return self.feature_combiner(i), self.targets[i]
