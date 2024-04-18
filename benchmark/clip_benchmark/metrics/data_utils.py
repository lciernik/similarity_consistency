import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Featurizer(torch.nn.Module):
    def __init__(self, model, normalize=True):
        super().__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, input):
        image_features = self.model.encode_image(input)
        if self.normalize:
            image_features = F.normalize(image_features, dim=-1)
        return image_features


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


def feature_extraction(featurizer, train_dataloader, dataloader, feature_dir, device, autocast, val_dataloader=None):
    """
    Extract features from the dataset using the featurizer model and store them in the feature_dir
    """
    # now we have to cache the features
    devices = [x for x in range(torch.cuda.device_count())]
    featurizer = torch.nn.DataParallel(featurizer, device_ids=devices)

    splits = ["_train", "_val", "_test"]
    for save_str, loader in zip(splits, [train_dataloader, val_dataloader, dataloader]):
        if loader is None:
            continue
        features = []
        targets = []
        num_batches_tracked = 0
        num_cached = 0
        with torch.no_grad():
            for images, target in tqdm(loader):
                images = images.to(device)

                with autocast():
                    feature = featurizer(images)

                features.append(feature.cpu())
                targets.append(target)

                num_batches_tracked += 1
                if (num_batches_tracked % 100) == 0:
                    features = torch.cat(features)
                    targets = torch.cat(targets)

                    torch.save(features, os.path.join(feature_dir, f'features{save_str}_cache_{num_cached}.pt'))
                    torch.save(targets, os.path.join(feature_dir, f'targets{save_str}_cache_{num_cached}.pt'))
                    num_cached += 1
                    features = []
                    targets = []

        if len(features) > 0:
            features = torch.cat(features)
            targets = torch.cat(targets)
            torch.save(features, os.path.join(feature_dir, f'features{save_str}_cache_{num_cached}.pt'))
            torch.save(targets, os.path.join(feature_dir, f'targets{save_str}_cache_{num_cached}.pt'))
            num_cached += 1

        features = torch.load(os.path.join(feature_dir, f'features{save_str}_cache_0.pt'))
        targets = torch.load(os.path.join(feature_dir, f'targets{save_str}_cache_0.pt'))
        for k in range(1, num_cached):
            next_features = torch.load(os.path.join(feature_dir, f'features{save_str}_cache_{k}.pt'))
            next_targets = torch.load(os.path.join(feature_dir, f'targets{save_str}_cache_{k}.pt'))
            features = torch.cat((features, next_features))
            targets = torch.cat((targets, next_targets))

        for k in range(num_cached):
            os.remove(os.path.join(feature_dir, f'features{save_str}_cache_{k}.pt'))
            os.remove(os.path.join(feature_dir, f'targets{save_str}_cache_{k}.pt'))

        torch.save(features, os.path.join(feature_dir, f'features{save_str}.pt'))
        torch.save(targets, os.path.join(feature_dir, f'targets{save_str}.pt'))


def get_fewshot_indices(features, targets, fewshot_k):
    """
    Get the indices of the features that are use for training the linear probe
    """
    length = len(features)
    perm = [p.item() for p in torch.randperm(length)]
    idxs = []
    counts = {}
    num_classes = 0

    for p in perm:
        target = targets[p].item()
        if target not in counts:
            counts[target] = 0
            num_classes += 1

        if fewshot_k < 0 or counts[target] < fewshot_k:
            counts[target] += 1
            idxs.append(p)

    for c in counts:
        if fewshot_k > 0 and counts[c] != fewshot_k:

            raise ValueError(f'insufficient data for eval with {fewshot_k} samples per class, '
                             f'only {counts[c]} samples for class {c}')

    return idxs


def get_feature_dl(feature_dir, batch_size, num_workers, fewshot_k, val_dataloader):
    """
    Load the features from the feature_dir and return the dataloaders for training, validation, and testing
    """
    features = torch.load(os.path.join(feature_dir, 'features_train.pt'))
    targets = torch.load(os.path.join(feature_dir, 'targets_train.pt'))

    idxs = get_fewshot_indices(features, targets, fewshot_k)
    train_features = features[idxs]
    train_labels = targets[idxs]
    if val_dataloader is not None:
        features_val = torch.load(os.path.join(feature_dir, 'features_val.pt'))
        targets_val = torch.load(os.path.join(feature_dir, 'targets_val.pt'))
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
    features_test = torch.load(os.path.join(feature_dir, 'features_test.pt'))
    targets_test = torch.load(os.path.join(feature_dir, 'targets_test.pt'))
    feature_test_dset = FeatureDataset(features_test, targets_test)
    feature_test_loader = DataLoader(
        feature_test_dset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True,
    )
    return feature_train_loader, feature_val_loader, feature_train_val_loader, feature_test_loader


def get_combined_feature_dl(feature_dirs, batch_size, num_workers, fewshot_k, feature_combiner_cls, use_val_ds, normalize):

    list_features = [torch.load(os.path.join(feature_dir, 'features_train.pt')) for feature_dir in feature_dirs]
    targets = torch.load(os.path.join(feature_dirs[0], 'targets_train.pt'))

    assert all([len(feat) == len(list_features[0]) for feat in list_features])

    idxs = get_fewshot_indices(list_features[0], targets, fewshot_k)

    list_train_features = [features[idxs] for features in list_features]
    train_labels = targets[idxs]

    feature_combiner_train = feature_combiner_cls()
    feature_train_dset = CombinedFeaturesDataset(list_train_features, train_labels, feature_combiner_train, normalize)
    feature_train_loader = DataLoader(feature_train_dset, batch_size=batch_size,
                                      shuffle=True, num_workers=num_workers, pin_memory=True, )

    if use_val_ds:
        list_features_val = [torch.load(os.path.join(feature_dir, 'features_val.pt')) for feature_dir in feature_dirs]
        targets_val = torch.load(os.path.join(feature_dirs[0], 'targets_val.pt'))

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

    list_features_test = [torch.load(os.path.join(feature_dir, 'features_test.pt')) for feature_dir in feature_dirs]
    targets_test = torch.load(os.path.join(feature_dirs[0], 'targets_test.pt'))

    feature_combiner_test = feature_combiner_cls(reference_combiner=feature_combiner_train)
    feature_test_dset = CombinedFeaturesDataset(list_features_test, targets_test, feature_combiner_test, normalize)
    feature_test_loader = DataLoader(
        feature_test_dset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True,
    )

    return feature_train_loader, feature_val_loader, feature_train_val_loader, feature_test_loader