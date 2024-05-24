import os
from typing import List, Union, Dict, Optional

import torch

from clip_benchmark.data.builder import get_dataset_collection_from_file, get_dataset_collection


def as_list(l):
    if not l:
        return []
    return [l] if type(l) != list else l


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


def get_list_of_datasets(base):
    datasets = []
    dataset_collection = get_dataset_collection()
    for name in as_list(base.dataset):
        if os.path.isfile(name):
            # If path, read file, each line is a dataset name
            datasets.extend(get_dataset_collection_from_file(name))
        elif name in dataset_collection:
            # if part of `dataset_collection`, retrieve from it
            datasets.extend(dataset_collection[name])
        else:
            # if not, assume it is simply the name of the dataset
            datasets.append(name)
    return datasets


def prepare_ds_name(dataset: str) -> str:
    # if dataset.startswith("wds/"):
    #     dataset = dataset.replace("wds/", "", 1)
    dataset = dataset.replace("/", "_")
    return dataset


def single_option_to_multiple_datasets(cur_option: List[str], datasets: List[str], name: str) -> List[str]:
    cur_len = len(cur_option)
    ds_len = len(datasets)
    if cur_len != ds_len:
        # If user wants to use same value for all datasets
        if cur_len == 1:
            return [cur_option[0]] * ds_len
        else:
            raise ValueError(f"The incommensurable number of {name}")
    else:
        return cur_option


def get_train_val_splits(
        train_split: Union[str, List[str]],
        val_proportion: Union[float, List[float]],
        datasets: List[str]
) -> Dict[str, Dict[str, Optional[Union[str, float]]]]:
    train_splits = as_list(train_split)
    train_splits = single_option_to_multiple_datasets(train_splits, datasets, "train_split")
    proportions = None
    if val_proportion is not None:
        proportions = as_list(val_proportion)
        proportions = single_option_to_multiple_datasets(proportions, datasets, "val_proportion")

    dataset_info = {}
    for i in range(len(datasets)):
        dataset_info[datasets[i]] = {
            "train_split": train_splits[i],
            "proportion": proportions[i] if proportions is not None else None
        }
    return dataset_info



def world_info_from_env():
    # from openclip
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size


def all_paths_exist(list_of_paths: List[str])->bool:
    return all([os.path.exists(p) for p in list_of_paths])
