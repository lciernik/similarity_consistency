import os

import torch
from tqdm import tqdm


def feature_extraction(featurizer, train_dataloader, eval_dataloader, feature_dir, device, autocast, val_dataloader=None):
    """
    Extract features from the dataset using the featurizer model and store them in the feature_dir
    """
    # now we have to cache the features
    devices = [x for x in range(torch.cuda.device_count())]
    featurizer = torch.nn.DataParallel(featurizer, device_ids=devices)

    splits = ["_train", "_val", "_test"]
    for save_str, loader in zip(splits, [train_dataloader, val_dataloader, eval_dataloader]):
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


def get_fewshot_indices(targets, fewshot_k):
    """
    Get the indices of the features that are use for training the linear probe
    """
    length = len(targets)
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
