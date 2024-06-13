from torch.utils.data import DataLoader

from clip_benchmark.data.data_utils import get_fewshot_indices
from clip_benchmark.data.feature_datasets import FeatureDataset, CombinedFeaturesDataset
from clip_benchmark.utils.utils import load_features_targets


def get_feature_dl(feature_dir: str, batch_size: int, num_workers: int, fewshot_k: int, idxs=None, load_train=True):
    """
    Load the features from the feature_dir and return the dataloaders for training, validation, and testing
    """
    features_test, targets_test = load_features_targets(feature_dir, split='test')
    feature_test_dset = FeatureDataset(features_test, targets_test)
    feature_test_loader = DataLoader(
        feature_test_dset, batch_size=batch_size,
        shuffle=False, num_workers=0,
        pin_memory=True,
    )

    if load_train:
        features, targets = load_features_targets(feature_dir, split='train')
        if fewshot_k < 0:  # if fewshot_k is -1, use the whole dataset
            train_features = features
            train_labels = targets
        else:
            if idxs is None:
                idxs = get_fewshot_indices(targets, fewshot_k)

            train_features = features[idxs]
            train_labels = targets[idxs]

        feature_train_dset = FeatureDataset(train_features, train_labels)
        feature_train_loader = DataLoader(feature_train_dset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers,
                                          pin_memory=True,
                                          )

    else:
        feature_train_loader = None

    return feature_train_loader, feature_test_loader


def get_combined_feature_dl(feature_dirs, batch_size, num_workers, fewshot_k, feature_combiner_cls,
                            normalize, load_train=True):
    if load_train:
        list_features, targets = load_features_targets(feature_dirs, split='train')

        if not all([len(feat) == len(list_features[0]) for feat in list_features]):
            raise ValueError("Features of the different models have different number of samples.")

        if fewshot_k < 0:  # if fewshot_k is -1, use the whole dataset
            list_train_features = list_features
            train_labels = targets
        else:
            idxs = get_fewshot_indices(targets, fewshot_k)

            list_train_features = [features[idxs] for features in list_features]
            train_labels = targets[idxs]

        feature_combiner_train = feature_combiner_cls()
        feature_train_dset = CombinedFeaturesDataset(list_train_features, train_labels, feature_combiner_train,
                                                     normalize)
        feature_train_loader = DataLoader(feature_train_dset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers, pin_memory=True, )
    else:
        # TODO: Load a trained feature combiner, if neccessary
        feature_train_loader = None
        feature_combiner_train = None

    list_features_test, targets_test = load_features_targets(feature_dirs, split='test')

    feature_combiner_test = feature_combiner_cls(reference_combiner=feature_combiner_train)
    feature_test_dset = CombinedFeaturesDataset(list_features_test, targets_test, feature_combiner_test, normalize)
    feature_test_loader = DataLoader(
        feature_test_dset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True,
    )

    return feature_train_loader, feature_test_loader
