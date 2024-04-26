import torch
import os


def _load_features(feature_root, model_id, split='train'):
    features = torch.load(os.path.join(feature_root, model_id, f'features_{split}.pt'))
    return features


def _check_models(feature_root, model_ids, split):
    prev_model_ids = model_ids

    model_ids = sorted(
        [mid for mid in model_ids if os.path.exists(os.path.join(feature_root, mid, f'features_{split}.pt'))])

    if len(set(prev_model_ids)) != len(set(model_ids)):
        print(f"Features do not exist for the following models: {set(prev_model_ids) - set(model_ids)}")
        print(f"Removing the above models from the list of models for distance computation.")

    # Check if enough remaining models to compute distance matrix
    assert len(model_ids) > 1, f"At least two models are required for distance computation"

    return model_ids
