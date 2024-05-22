from scripts.helper import load_models, prepare_for_combined_usage
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import SpectralClustering, AffinityPropagation
import os

corr_method = "spearman"
rsa_mat = torch.load(
    f"/home/space/diverse_priors/model_similarities/rsa_correlation_{corr_method}/imagenet_subset_10k/rsa_distance_matrix.pt")

dataset = 'imagenet-subset-10k'
models, n_models = load_models('./scripts/models_config.json')
num_clusters = 4

model_names, sources, model_parameters, module_names = prepare_for_combined_usage(models)
models = zip(model_names, sources, model_parameters, module_names)


def _get_model_id(model, model_parameters):
    if not model_parameters:
        return model
    model_slug = model
    model_suffix = model_parameters.get("variant", "")
    if model_suffix:
        model_slug = f"{model_slug}_{model_suffix}"
    model_suffix = model_parameters.get("dataset", "")
    if model_suffix:
        model_slug = f"{model_slug}_{model_suffix}"
    return model_slug


# Get model ids
model_ids = [_get_model_id(model[0], model[2]) for model in models]
model_ids = [(model_id + '-' + dataset).replace('/', '_') for model_id in model_ids]
model_ids = sorted(model_ids)
model_ids = [mid.replace(f"-{dataset}", "") for mid in model_ids]

# compute affinity mat
df = pd.DataFrame(rsa_mat.numpy(), index=model_ids, columns=model_ids)
affinity_mat = df.copy().abs()
np.fill_diagonal(affinity_mat.values, 1)

clustering = SpectralClustering(n_clusters=num_clusters,
                                affinity='precomputed',
                                assign_labels='discretize',
                                random_state=0)

labels = clustering.fit_predict(affinity_mat.values, y=None)
