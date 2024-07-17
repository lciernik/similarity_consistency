import argparse
import json
import os
import random
import sqlite3
import warnings
from itertools import product
from pathlib import Path
from typing import Any
from typing import List, Union, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from clip_benchmark.data.builder import get_dataset_collection_from_file, get_dataset_collection
from clip_benchmark.data.constants import probe_dataset_map


def as_list(l):
    if not l:
        return []
    return [l] if type(l) != list else l


## Load features and targets
def load_features(feature_root: str, model_id: Optional[str] = None, split: str = 'train') -> torch.Tensor:
    model_dir = os.path.join(feature_root, model_id) if model_id else feature_root
    features = torch.load(os.path.join(model_dir, f'features_{split}.pt'))
    return features


def load_targets(feature_root: str, model_id: Optional[str] = None, split: str = 'train') -> torch.Tensor:
    model_dir = os.path.join(feature_root, model_id) if model_id else feature_root
    targets = torch.load(os.path.join(model_dir, f'targets_{split}.pt'))
    return targets


def check_equal_targets(list_targets: List[torch.Tensor]) -> bool:
    if len(list_targets) > 1:
        first_targets = list_targets[0]
        for curr_target in list_targets[1:]:
            if not (first_targets == curr_target).all().item():
                return False
    return True


def load_features_targets(
        feature_root: str,
        model_id: Optional[str] = None,
        split: str = 'train'
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(feature_root, list):
        features = [load_features(f, model_id, split) for f in feature_root]
        targets = [load_targets(f, model_id, split) for f in feature_root]
        if not check_equal_targets(targets):
            raise ValueError("Not all targets are equal.")
        targets = targets[0]
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


def map_to_probe_dataset(dataset: str, verbose: bool = False) -> str:
    if dataset in probe_dataset_map:
        if verbose:
            print(f"Dataset mapping for loading probes found. Mapping {dataset} to {probe_dataset_map[dataset]}")
        return probe_dataset_map[dataset]
    return dataset


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


def all_paths_exist(list_of_paths: List[str]) -> bool:
    return all([os.path.exists(p) for p in list_of_paths])


def set_all_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)


def prepare_device(distributed: bool) -> str:
    if torch.cuda.is_available():
        if distributed:
            local_rank, rank, world_size = world_info_from_env()
            device = 'cuda:%d' % local_rank
            torch.cuda.set_device(device)
        else:
            device = "cuda"
        return device
    else:
        return "cpu"


def get_combination(
        fewshot_ks: List[int],
        fewshot_lrs: List[float],
        fewshot_epochs: List[int],
        seeds: List[int],
        weight_decays: List[float],
        weight_decay_types: List[str],
) -> Tuple[int, float, int, int, float, str]:
    combs = []
    combs.extend(
        list(
            product(
                fewshot_ks,
                fewshot_lrs,
                fewshot_epochs,
                seeds,
                weight_decays,
                weight_decay_types,
            )
        )
    )
    return combs[int(os.environ["SLURM_ARRAY_TASK_ID"])]


def get_list_of_models(base: argparse.Namespace) -> List[Tuple[str, str, dict, str, str, str]]:
    """Get list of models and config to evaluate."""
    models = as_list(base.model)
    srcs = as_list(base.model_source)
    params = as_list(base.model_parameters)
    module_names = as_list(base.module_name)
    feature_alignments = as_list(base.feature_alignment)
    model_keys = as_list(base.model_key)

    assert len(models) == len(srcs), "The number of model_source should be the same as the number of models"
    assert len(models) == len(params), "The number of model_parameters should be the same as the number of models"
    assert len(models) == len(module_names), "The number of module_name should be the same as the number of models"
    assert len(models) == len(
        feature_alignments), "The number of feature_alignment should be the same as the number of models"
    assert len(models) == len(model_keys), "The number of model_key should be the same as the number of models"

    models_w_config = list(zip(models, srcs, params, module_names, feature_alignments, model_keys))
    models_w_config = sorted(models_w_config, key=lambda x: x[-1])
    return models_w_config


def make_results_df(exp_args: argparse.Namespace, model_ids: List[str], metrics: Dict[str, float]) -> pd.DataFrame:
    results_current_run = pd.DataFrame(index=range(1))

    # experiment config
    results_current_run["task"] = exp_args.task
    results_current_run["mode"] = exp_args.mode
    results_current_run["combiner"] = exp_args.feature_combiner \
        if exp_args.task == 'linear_probe' and exp_args.mode == 'combined_models' \
        else None
    # dataset
    results_current_run["dataset"] = exp_args.dataset
    results_current_run["feature_normalization"] = exp_args.normalize
    results_current_run["feature_alignment"] = json.dumps(exp_args.feature_alignment)
    results_current_run["train_split"] = exp_args.train_split
    results_current_run["val_proportion"] = exp_args.val_proportion
    results_current_run["test_split"] = exp_args.split
    # model(s)
    results_current_run["model_ids"] = json.dumps(model_ids)
    results_current_run["model"] = json.dumps(exp_args.model)
    results_current_run["model_source"] = json.dumps(exp_args.model_source)
    results_current_run["model_parameters"] = json.dumps(exp_args.model_parameters)
    results_current_run["module_name"] = json.dumps(exp_args.module_name)
    # hyperparameters
    results_current_run["fewshot_k"] = exp_args.fewshot_k
    results_current_run["fewshot_lr"] = exp_args.fewshot_lr
    results_current_run["fewshot_epochs"] = exp_args.fewshot_epochs
    results_current_run["batch_size"] = exp_args.batch_size
    results_current_run["seed"] = exp_args.seed
    results_current_run["weight_decay"] = exp_args.weight_decay
    results_current_run["weight_decay_type"] = exp_args.weight_decay_type

    # metrics
    def flatten_metrics(curr_metrics):
        new_metrics = {}
        if 'train_metrics' in curr_metrics:
            new_metrics.update({f'train_{k}': v for k, v in curr_metrics['train_metrics'].items()})
        elif 'test_metrics' in curr_metrics:
            # We get an Error when the Database has different Columns than the current run
            new_metrics.update({f'train_{k}': None for k, v in curr_metrics['test_metrics'].items()})
        else:
            warnings.warn(
                "No train or test metrics found in the metrics dictionary. Maybe Addition to Database wont work")
        if 'test_metrics' in curr_metrics:
            new_metrics.update({f'test_{k}': v for k, v in curr_metrics['test_metrics'].items()})
        new_metrics.update({k: v for k, v in curr_metrics.items() if not isinstance(v, dict)})
        return new_metrics

    flattened_metrics = flatten_metrics(metrics)
    for key, value in flattened_metrics.items():
        if key in results_current_run:
            continue
        results_current_run[key] = value

    # # serialize object columns
    # for col in results_current_run:
    #     if results_current_run[col].dtype == "object":
    #         try:
    #             results_current_run[col] = results_current_run[col].apply(json.dumps)
    #         except TypeError as e:
    #             print(col)
    #             print(results_current_run[col])
    #             raise e

    return results_current_run


def save_results(args: argparse.Namespace, model_ids: List[str], metrics: Dict[str, float],
                 out_path: str) -> None:
    """Save the results to json file."""
    results_current_run = make_results_df(exp_args=args, model_ids=model_ids, metrics=metrics)

    if len(results_current_run) == 0:
        raise ValueError("results_current_run had no entries")

    results_current_run.to_json(os.path.join(out_path, "results.json"))


def get_base_evaluator_args(
        args: argparse.Namespace,
        feature_dirs: List[str],
        model_dirs: List[str],
        predictions_dir: str
) -> Dict[str, Any]:
    base_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "lr": args.fewshot_lr,
                   "epochs": args.fewshot_epochs, "seed": args.seed, "device": args.device,
                   "fewshot_k": args.fewshot_k, "feature_dirs": feature_dirs, "model_dirs": model_dirs,
                   "predictions_dir": predictions_dir, "normalize": args.normalize,
                   "val_proportion": args.val_proportion, "weight_decay": args.weight_decay,
                   "weight_decay_type": args.weight_decay_type}
    return base_kwargs


def retrieve_model_dataset_results(base_path_exp: str, verbose: Optional[bool] = False) -> pd.DataFrame:
    path = Path(base_path_exp)
    dfs = []
    for fn in path.rglob("**/results.json"):
        df = pd.read_json(fn)
        dfs.append(df)

    if len(dfs) == 0:
        # backward compatibility
        bak_fn = path / 'results.db'
        if bak_fn.is_file():
            print(f'Did not find any results.json files. Trying to load data from {bak_fn}')
            try:
                conn = sqlite3.connect(bak_fn)
                df = pd.read_sql('SELECT * FROM "results"', conn)
                conn.close()
            except pd.errors.DatabaseError as e:
                print(f"Tried to extract data from {path=}, but got Error: {e}")
                raise e

            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(json.loads)
        else:
            raise FileNotFoundError(f"No results found for in {base_path_exp=}")
    else:
        df = pd.concat(dfs).reset_index(drop=True)

    if verbose:
        print(f"Found {len(df)} results in {base_path_exp=}")
    return df
