import numpy as np
from torch.utils.data import DataLoader

from clip_benchmark.data.data_utils import get_fewshot_indices
from clip_benchmark.data.feature_datasets import FeatureDataset, CombinedFeaturesDataset
from clip_benchmark.utils.utils import load_features_targets


def get_feature_dl(feature_dir, batch_size, num_workers, fewshot_k, use_val_ds, idxs=None):
    """
    Load the features from the feature_dir and return the dataloaders for training, validation, and testing
    """
    features, targets = load_features_targets(feature_dir, split='train')

    if idxs is None:
        idxs = get_fewshot_indices(targets, fewshot_k)
    train_features = features[idxs]
    train_labels = targets[idxs]

    if use_val_ds:
        # TODO: adapt this to fewshot_k setting when using val_ds
        features_val, targets_val = load_features_targets(feature_dir, split='val')
        feature_val_dset = FeatureDataset(features_val, targets_val)
        feature_val_loader = DataLoader(
            feature_val_dset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            pin_memory=True,
        )
        feature_train_val_dset = FeatureDataset(np.concatenate((train_features, features_val)),
                                                np.concatenate((train_labels, targets_val)))
        feature_train_val_loader = DataLoader(
            feature_train_val_dset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            pin_memory=True,
        )
    else:
        feature_val_loader = None
        feature_train_val_loader = None
    feature_train_dset = FeatureDataset(train_features, train_labels)
    feature_train_loader = DataLoader(feature_train_dset, batch_size=batch_size,
                                      shuffle=True, num_workers=num_workers,
                                      pin_memory=True,
                                      )
    features_test, targets_test = load_features_targets(feature_dir, split='test')
    feature_test_dset = FeatureDataset(features_test, targets_test)
    feature_test_loader = DataLoader(
        feature_test_dset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    return feature_train_loader, feature_val_loader, feature_train_val_loader, feature_test_loader


def get_combined_feature_dl(feature_dirs, batch_size, num_workers, fewshot_k, feature_combiner_cls, use_val_ds,
                            normalize):
    list_features, targets = load_features_targets(feature_dirs, split='train')

    if not all([len(feat) == len(list_features[0]) for feat in list_features]):
        raise ValueError("Features of the different models have different number of samples.")

    idxs = get_fewshot_indices(targets, fewshot_k)

    list_train_features = [features[idxs] for features in list_features]
    train_labels = targets[idxs]

    feature_combiner_train = feature_combiner_cls()
    feature_train_dset = CombinedFeaturesDataset(list_train_features, train_labels, feature_combiner_train, normalize)
    feature_train_loader = DataLoader(feature_train_dset, batch_size=batch_size,
                                      shuffle=True, num_workers=num_workers, pin_memory=True, )

    if use_val_ds:
        list_features_val, targets_val = load_features_targets(feature_dirs, split='val')

        feature_combiner_val = feature_combiner_cls(reference_combiner=feature_combiner_train)
        feature_val_dset = CombinedFeaturesDataset(list_features_val,
                                                   targets_val,
                                                   feature_combiner_val,
                                                   normalize)
        feature_val_loader = DataLoader(
            feature_val_dset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            pin_memory=True,
        )
        list_train_val_features = [np.concatenate((feat_train, feat_val)) for feat_train, feat_val in
                                   zip(list_train_features, list_features_val)]

        feature_combiner_train = feature_combiner_cls()
        feature_train_val_dset = CombinedFeaturesDataset(list_train_val_features,
                                                         np.concatenate((train_labels, targets_val)),
                                                         feature_combiner_train,
                                                         normalize)
        feature_train_val_loader = DataLoader(
            feature_train_val_dset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            pin_memory=True,
        )
    else:
        feature_val_loader = None
        feature_train_val_loader = None
    list_features_test, targets_test = load_features_targets(feature_dirs, split='test')

    feature_combiner_test = feature_combiner_cls(reference_combiner=feature_combiner_train)
    feature_test_dset = CombinedFeaturesDataset(list_features_test, targets_test, feature_combiner_test, normalize)
    feature_test_loader = DataLoader(
        feature_test_dset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True,
    )

    return feature_train_loader, feature_val_loader, feature_train_val_loader, feature_test_loader
