import os

import torch

## Load features and targets
def load_features(feature_root, model_id=None, split='train'):
    model_dir = os.path.join(feature_root, model_id) if model_id else feature_root
    features = torch.load(os.path.join(model_dir, f'features_{split}.pt'))
    return features


def load_targets(feature_root, model_id=None, split='train'):
    model_dir = os.path.join(feature_root, model_id) if model_id else feature_root
    targets = torch.load(os.path.join(model_dir, f'targets_{split}.pt'))
    return targets


def load_features_targets(feature_root, model_id=None, split='train'):
    if isinstance(feature_root, list):
        features = [load_features(f, model_id, split) for f in feature_root]
        targets = load_targets(feature_root[0], model_id, split)
    else:
        features = load_features(feature_root, model_id, split)
        targets = load_targets(feature_root, model_id, split)
    return features, targets


## Check if features exist for all models
def check_models(feature_root, model_ids, split):
    prev_model_ids = model_ids

    model_ids = sorted(
        [mid for mid in model_ids if os.path.exists(os.path.join(feature_root, mid, f'features_{split}.pt'))])

    if len(set(prev_model_ids)) != len(set(model_ids)):
        print(f"Features do not exist for the following models: {set(prev_model_ids) - set(model_ids)}")
        print(f"Removing the above models from the list of models for distance computation.")

    # Check if enough remaining models to compute distance matrix
    assert len(model_ids) > 1, f"At least two models are required for distance computation"

    return model_ids
