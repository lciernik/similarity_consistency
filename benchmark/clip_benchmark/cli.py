"""Console script for clip_benchmark."""
import argparse
import csv
import json
import os
import random
import sqlite3
import sys
from copy import copy
from itertools import product, combinations
from typing import List, Tuple, Union, Dict

import numpy as np
import pandas as pd
import torch

from clip_benchmark.data import (build_dataset, get_dataset_collate_fn, get_feature_combiner_cls)
from clip_benchmark.models import load_model
from clip_benchmark.tasks import compute_sim_matrix
from clip_benchmark.tasks.linear_probe import SingleModelEvaluator, CombinedModelEvaluator, \
    EnsembleModelEvaluator
from clip_benchmark.utils.utils import as_list, get_list_of_datasets


def get_parser_args() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Get the parser arguments."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_eval = subparsers.add_parser('eval', help='Evaluate')
    # DATASET
    parser_eval.add_argument('--dataset', type=str, default="cifar10", nargs="+",
                             help="Dataset(s) to use for the benchmark. Can be the name of a dataset, or a collection "
                                  "name ('vtab', 'vtab+', 'imagenet_robustness', 'retrieval') or path of a text file "
                                  "where each line is a dataset name")
    parser_eval.add_argument('--dataset_root', default="root", type=str,
                             help="dataset root folder where the data are downloaded. Can be in the form of a "
                                  "template depending on dataset name, e.g., --dataset_root='data/{dataset}'. "
                                  "This is useful if you evaluate on multiple data.")
    parser_eval.add_argument('--split', type=str, default="test", help="Dataset split to use")
    parser_eval.add_argument('--test_split', dest="split", action='store', type=str, default="test",
                             help="Dataset split to use")
    parser_eval.add_argument('--train_split', type=str, nargs='+', default="train",
                             help="Dataset(s) train split names")
    mutually_exclusive = parser_eval.add_mutually_exclusive_group()
    mutually_exclusive.add_argument('--val_split', default=None, type=str, nargs="+",
                                    help="Dataset(s) validation split names. Mutually exclusive with val_proportion.")
    mutually_exclusive.add_argument('--val_proportion', default=None, type=float, nargs="+",
                                    help="what is the share of the train dataset will be used for validation part, "
                                         "if it doesn't predefined. Mutually exclusive with val_split")
    parser_eval.add_argument('--wds_cache_dir', default=None, type=str,
                             help="optional cache directory for webdataset only")
    parser_eval.add_argument('--custom_classname_file', default=None, type=str,
                             help="use custom json file with classnames for each dataset, where keys are dataset "
                                  "names and values are list of classnames.")

    # FEATURES
    parser_eval.add_argument('--feature_root', default="features", type=str,
                             help="feature root folder where the features are stored.")
    # TODO: change alignment to argument such that it can be model specific, b/c some model do not have alignment.
    parser_eval.add_argument('--feature_alignment', nargs='?', const='gLocal',
                             type=lambda x: None if x == '' else x)
    parser_eval.add_argument('--normalize', dest='normalize', action="store_true", default=True,
                             help="enable features normalization")
    parser_eval.add_argument('--no-normalize', dest='normalize', action='store_false',
                             help="disable features normalization")

    # MODEL(S)
    parser_eval.add_argument('--model', type=str, nargs="+", default=["dinov2-vit-large-p14"],
                             help="Thingsvision model string")
    parser_eval.add_argument('--model_source', type=str, nargs="+", default=["ssl"],
                             help="For each model, indicate the source of the model. "
                                  "See thingsvision for more details.")
    parser_eval.add_argument('--model_parameters', nargs="+", type=str,
                             help='A serialized JSON dictionary of key-value pairs.')
    parser_eval.add_argument('--module_name', type=str, nargs="+", default=["norm"], help="Module name")

    # TASKS
    parser_eval.add_argument('--task', type=str, default="linear_probe",
                             choices=["linear_probe", "model_similarity"],
                             help="Task to evaluate on. With --task=auto, the task is automatically inferred from the "
                                  "dataset.")
    parser_eval.add_argument('--mode', type=str, default="single_model",
                             choices=["single_model", "combined_models", "ensemble"],
                             help="Mode to use for linear probe task.")
    parser_eval.add_argument('--eval_combined', action="store_true",
                             help="Whether the features of the different models should be used in combined fashion.")
    parser_eval.add_argument('--feature_combiner', type=str, default="concat",
                             choices=['concat', 'concat_pca'], help="Feature combiner to use")

    parser_eval.add_argument('--fewshot_k', default=[-1], type=int, nargs="+",
                             help="for linear probe, how many shots. -1 = whole dataset.")
    parser_eval.add_argument('--fewshot_epochs', default=[10], type=int, nargs='+',
                             help="for linear probe, how many epochs.")
    parser_eval.add_argument('--fewshot_lr', default=[0.1], type=float, nargs='+',
                             help="for linear probe, what is the learning rate.")
    parser_eval.add_argument('--batch_size', default=64, type=int)
    parser_eval.add_argument('--no_amp', action="store_false", dest="amp", default=True,
                             help="whether to use mixed precision")
    parser_eval.add_argument("--skip_load", action="store_true",
                             help="for linear probes, when everything is cached, no need to load model.")
    parser_eval.add_argument('--skip_existing', default=False, action="store_true",
                             help="whether to skip an evaluation if the output file exists.")

    ### Model similarity
    parser_eval.add_argument('--sim_method', type=str, default="cka",
                             choices=['cka', 'rsa'], help="Method to use for model similarity task.")
    parser_eval.add_argument('--sim_kernel', type=str, default="linear",
                             choices=['linear'], help="Kernel used during CKA. Ignored if sim_method is rsa.")
    parser_eval.add_argument('--rsa_method', type=str, default="correlation",
                             choices=['cosine', 'correlation'],
                             help="Method used during RSA. Ignored if sim_method is cka.")
    parser_eval.add_argument('--corr_method', type=str, default="spearman",
                             choices=['pearson', 'spearman'],
                             help="Kernel used during CKA. Ignored if sim_method is cka.")
    parser_eval.add_argument('--sigma', type=float, default=None, help="sigma for CKA rbf kernel.")
    parser_eval.add_argument('--biased_cka', action="store_false", dest="unbiased", help="use biased CKA")

    # STORAGE
    parser_eval.add_argument('--output_root', default="results", type=str,
                             help="Path to root folder where the results are stored.")
    parser_eval.add_argument('--model_root', default="models", type=str,
                             help="Path to root folder where linear probe model checkpoints are stored.")

    # GENERAL
    parser_eval.add_argument('--num_workers', default=4, type=int)

    parser_eval.add_argument("--distributed", action="store_true", help="evaluation in parallel")
    parser_eval.add_argument('--quiet', dest='verbose', action="store_false",
                             help="suppress verbose messages")

    # REPRODUCABILITY
    parser_eval.add_argument('--seed', default=[0], type=int, nargs='+', help="random seed.")

    parser_eval.set_defaults(which='eval')

    parser_build = subparsers.add_parser('build', help='Build CSV from evaluations')
    parser_build.add_argument('files', type=str, nargs="+", help="path(s) of JSON result files")
    parser_build.add_argument('--output', type=str, default="benchmark.csv", help="CSV output file")
    parser_build.set_defaults(which='build')

    args = parser.parse_args()
    return parser, args


def prepare_args(args: argparse.Namespace, model_info: Tuple[str, str, dict, str]) -> argparse.Namespace:
    args.model = model_info[0]  # model
    args.model_source = model_info[1]  # model_source
    args.model_parameters = model_info[2]  # model_parameters
    args.module_name = model_info[3]  # module_name
    return args


def prepare_combined_args(args: argparse.Namespace, model_comb: List[Tuple[str, str, dict, str]]) -> argparse.Namespace:
    args.model = [tup[0] for tup in model_comb]
    args.model_source = [tup[1] for tup in model_comb]
    args.model_parameters = [tup[2] for tup in model_comb]
    args.module_name = [tup[3] for tup in model_comb]
    return args


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
        fewshot_lrs: List[int],
        fewshot_epochs: List[int],
        seeds: List[float]
) -> Tuple[int, float, int, int]:
    combs = []
    combs.extend(
        list(
            product(
                fewshot_ks,
                fewshot_lrs,
                fewshot_epochs,
                seeds,
            )
        )
    )
    return combs[int(os.environ["SLURM_ARRAY_TASK_ID"])]


def get_list_of_models(base: argparse.Namespace) -> List[Tuple[str, str, dict, str]]:
    """Get list of models and config to evaluate."""
    models = as_list(base.model)
    srcs = as_list(base.model_source)
    params = as_list(base.model_parameters)
    params = [json.loads(x) for x in params]
    module_names = as_list(base.module_name)

    assert len(models) == len(srcs), "The number of model_source should be the same as the number of models"
    assert len(models) == len(params), "The number of model_parameters should be the same as the number of models"
    assert len(models) == len(module_names), "The number of module_name should be the same as the number of models"

    return list(zip(models, srcs, params, module_names))


def get_model_id(model: str, model_parameters: Union[dict, None]) -> str:
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


def prepare_ds_name(dataset: str) -> str:
    if dataset.startswith("wds/"):
        return dataset.replace("wds/", "", 1)
    else:
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


def get_train_val_splits(train_split, val_split, val_proportion, datasets):
    # TODO add typing
    train_splits = as_list(train_split)
    train_splits = single_option_to_multiple_datasets(train_splits, datasets, "train_split")
    proportions, val_splits = None, None
    if val_split is not None:
        val_splits = as_list(val_split)
        val_splits = single_option_to_multiple_datasets(val_splits, datasets, "val_split")
    if val_proportion is not None:
        proportions = as_list(val_proportion)
        proportions = single_option_to_multiple_datasets(proportions, datasets, "val_proportion")

    dataset_info = {}
    for i in range(len(datasets)):
        dataset_info[datasets[i]] = {
            "train_split": train_splits[i],
            "val_split": val_splits[i] if val_splits is not None else None,
            "proportion": proportions[i] if proportions is not None else None
        }
    return dataset_info


def get_hyperparams_name(args: argparse.Namespace) -> str:
    """Get the hyperparameters name for the output path."""
    fewshot_slug = "no_fewshot" if args.fewshot_k == -1 else f"fewshot_{args.fewshot_k}"
    subpath = os.path.join(fewshot_slug,
                           f"fewshot_lr_{args.fewshot_lr}",
                           f"fewshot_epochs_{args.fewshot_epochs}",
                           f"batch_size_{args.batch_size}",
                           f"seed_{args.seed:02d}",
                           )
    return subpath


def check_root_paths(args: argparse.Namespace) -> None:
    """Check existence of the feature, model and output folders."""
    if not os.path.exists(args.dataset_root):
        raise FileNotFoundError(f"Dataset root folder {args.dataset_root} does not exist.")
    if not os.path.exists(args.feature_root):
        raise FileNotFoundError(f"Feature root folder {args.feature_root} does not exist.")
    if not os.path.exists(args.model_root):
        raise FileNotFoundError(f"Model root folder {args.model_root} does not exist.")
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root, exist_ok=True)
        if args.verbose:
            print(f'Created path ({args.output_root}), where results are to be stored ...')


def make_paths(args: argparse.Namespace, dataset_name: str):
    check_root_paths(args)

    task, mode = args.task, args.mode

    dataset_slug = dataset_name.replace('/', '_')

    models = as_list(args.model)
    model_params = as_list(args.model_parameters)

    # Create model_ids based on the model and model_params
    # TODO: add aligned keyword to model_id
    # feature_alignment=args.feature_alignment if args.feature_alignment is not None else "no_alignment"
    model_ids = [get_model_id(model, model_params) for model, model_params in zip(models, model_params)]

    # Create list of feature directories for each dataset and model_ids.
    feature_dirs = [os.path.join(args.feature_root, dataset_slug, model_id) for model_id in model_ids]

    # Create list of model checkpoint directories (for the linear probe) for each dataset, model_id, and hyperparameter
    # combination
    hyperparams_slug = get_hyperparams_name(args)
    if task == "linear_probe" and mode == "combined_models":
        model_slug = '__'.join(model_ids) + f"_{args.feature_combiner}"
        model_dirs = [os.path.join(args.model_root, dataset_slug, model_slug, hyperparams_slug)]
    else:
        model_dirs = [os.path.join(args.model_root, dataset_slug, model_id, hyperparams_slug) for model_id in model_ids]

    # create output path based on the task, mode, dataset, (combined) model_ids
    # NOTE: In this folder we will store the results of different hyperparameter combinations in a results database.
    model_slug = '__'.join(model_ids)
    if task == "linear_probe" and mode == "combined_models":
        model_slug = model_slug + f"_{args.feature_combiner}"
    out_dir_root = os.path.join(args.output_root, task, mode, dataset_slug, model_slug)
    # TODO: remove out_dir_pred because we store the predictions.pkl in the model directory
    out_dir_pred = os.path.join(out_dir_root, hyperparams_slug)
    if not os.path.exists(out_dir_pred):
        os.makedirs(out_dir_pred, exist_ok=True)
        if args.verbose:
            print(f'Created path ({out_dir_root}), where results are to be stored ...')

    return feature_dirs, model_dirs, out_dir_root, out_dir_pred, model_ids


def make_results_df(exp_args: argparse.Namespace, model_ids: List[str], metrics: Dict[str, float]) -> pd.DataFrame:
    results_current_run = pd.DataFrame(index=range(1))

    # experiment config
    results_current_run["task"] = exp_args["task"]
    results_current_run["mode"] = exp_args["mode"]
    results_current_run["combiner"] = exp_args["combiner"]
    # dataset
    results_current_run["dataset"] = exp_args["dataset"]
    results_current_run["feature_normalization"] = exp_args["normalize"]
    results_current_run["feature_alignment"] = exp_args["feature_alignment"]
    results_current_run["train_split"] = exp_args["train_split"]
    results_current_run["val_split"] = exp_args["val_split"]
    results_current_run["test_split"] = exp_args["split"]
    # model(s)
    results_current_run["model_ids"] = model_ids
    results_current_run["model"] = exp_args["model"]
    results_current_run["model_source"] = exp_args["model_source"]
    results_current_run["model_parameters"] = exp_args["model_parameters"]
    results_current_run["module_name"] = exp_args["module_name"]
    # hyperparameters
    results_current_run["fewshot_k"] = exp_args["fewshot_k"]
    results_current_run["fewshot_lr"] = exp_args["fewshot_lr"]
    results_current_run["fewshot_epochs"] = exp_args["fewshot_epochs"]
    results_current_run["batch_size"] = exp_args["batch_size"]
    results_current_run["seed"] = exp_args["seed"]
    # metrics
    for key, value in metrics.items():
        # skip if the key already exists
        if key in results_current_run:
            continue
        results_current_run[key] = value

    # serialize object columns
    for col in results_current_run:
        if results_current_run[col].dtype == "object":
            try:
                results_current_run[col] = results_current_run[col].apply(json.dumps)
            except TypeError as e:
                print(col)
                print(results_current_run.loc[0, col])
                raise e

    return results_current_run


def save_results(args: argparse.Namespace, model_ids: List[str], metrics: Dict[str, float],
                 out_path: str) -> None:
    """Save the results to a database (created if not existant)."""
    results_current_run = make_results_df(exp_args=args, model_ids=model_ids, metrics=metrics)

    if len(results_current_run) == 0:
        raise ValueError("results_current_run had no entries")

    database_path = os.path.join(out_path, "results.db")
    conn = sqlite3.connect(database_path)
    results_current_run.to_sql("results", con=conn, index=False, if_exists="append")
    conn.close()


def main():
    parser, base = get_parser_args()
    if not hasattr(base, "which"):
        parser.print_help()
        return

    if base.which == "build":
        main_build(base)
    elif base.which == "eval":
        if base.mode == "model_similarity":
            main_model_sim(base)
        else:
            main_eval(base)


def main_build(base):
    # Build a benchmark single CSV file from a set of evaluations (JSON files)
    rows = []
    fieldnames = set()

    def process_file(fpath: str):
        data = json.load(open(fpath))
        row = {}
        row.update(data["metrics"])
        row.update(data)
        del row["metrics"]
        row['model_fullname'] = "__".join(row['model_ids'])
        for field in row.keys():
            fieldnames.add(field)
        rows.append(row)

    for path in base.files:
        if os.path.isdir(path):
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json")]
            for file in files:
                process_file(file)
        else:
            process_file(path)
    with open(base.output_root, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main_model_sim(base):
    base.device = prepare_device(base.distributed)

    # Get list of data to evaluate on
    datasets = get_list_of_datasets(base)

    # Get train and val splits
    dataset_info = get_train_val_splits(base.train_split, base.val_split, base.val_proportion, datasets)

    dataset = datasets[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    train_split = dataset_info[dataset]["train_split"]

    # Get list of models to evaluate
    models = get_list_of_models(base)

    # Get model ids
    model_ids = [get_model_id(model[0], model[2]) for model in models]
    model_ids = [(model_id + '-' + dataset).replace('/', '_') for model_id in model_ids]

    # Compute CKA matrix
    sim_matrix, model_ids = compute_sim_matrix(base.sim_method,
                                               base.feature_root,
                                               model_ids,
                                               train_split,
                                               kernel=base.sim_kernel,
                                               rsa_method=base.rsa_method,
                                               corr_method=base.corr_method,
                                               backend='torch',
                                               unbiased=base.unbiased,
                                               device=base.device,
                                               sigma=base.sigma,
                                               )
    # Save the similarity matrix
    if not os.path.exists(base.output_root):
        os.makedirs(base.output_root, exist_ok=True)
        if base.verbose:
            print(f'Created path ({base.output_root}), where results are to be stored ...')
    if base.sim_method == 'cka':
        sim_config_slug = f"cka_kernel_{base.sim_kernel}_unbiased_{base.unbiased}_sigma_{base.sigma}"
    else:
        sim_config_slug = f"rsa_method_{base.rsa_method}_corr_method_{base.corr_method}"

    out_res = os.path.join(base.output_root, f'{sim_config_slug}_similarity_matrix.pt')
    if base.verbose:
        print(f"Dump {base.sim_method.upper()} matrix to: {out_res}")
    torch.save(sim_matrix, out_res)
    with open(os.path.join(base.output_root, f'{sim_config_slug}_model_ids.txt'), "w") as file:
        for string in model_ids:
            file.write(string + "\n")

    return 0


def main_eval(base):
    # prepare run combinations
    (fewshot_k, fewshot_lr, fewshot_epochs, rnd_seed) = get_combination(
        base.fewshot_k,
        base.fewshot_lr,
        base.fewshot_epochs,
        base.seed
    )
    # Get list of models to evaluate
    models = get_list_of_models(base)

    # Get list of data to evaluate on
    datasets = get_list_of_datasets(base)

    # Get train and val splits
    dataset_info = get_train_val_splits(base.train_split, base.val_split, base.val_proportion, datasets)

    if base.verbose:
        print(f"Models: {models}")
        print(f"Datasets: {datasets}")

    if base.eval_combined:
        # TODO: implement different ways how to select the model combinations
        # Now assumption that passed models are combined together (all permutations)
        n_models = len(models)
        model_combinations = []
        for i in range(2, min(n_models + 1, 11)):
            # TODO this is only for fast testing till we find better combinations
            model_combinations += list(combinations(models, i))[:10]

        runs = product(model_combinations, datasets)
        arg_fn = prepare_combined_args
    else:
        runs = product(models, datasets)
        arg_fn = prepare_args

    if base.distributed:
        local_rank, rank, world_size = world_info_from_env()
        runs = list(runs)
        random.seed(base.seed)
        random.shuffle(runs)
        runs = [r for i, r in enumerate(runs) if i % world_size == rank]

    # seed random number generator (important for reproducibility of results)
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)

    for model_info, dataset in runs:

        args = copy(base)
        args = arg_fn(args, model_info)
        args.dataset = dataset
        args.train_split = dataset_info[dataset]["train_split"]
        args.val_split = dataset_info[dataset]["val_split"]
        args.val_proportion = dataset_info[dataset]["proportion"]
        args.fewshot_k = fewshot_k
        args.fewshot_lr = fewshot_lr
        args.fewshot_epochs = fewshot_epochs
        args.seed = rnd_seed
        args.task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

        try:
            run(args)
        except Exception as e:
            print(
                f"An error occurred during the run with: "
                f"{model_info} and {dataset}. "
                f"Continuing with the next run.",
                flush=True)
            print(e, flush=True)
            raise e


def get_extraction_model_n_dataloader(args, dataset_root, task):
    if args.skip_load or isinstance(args.model, list):
        model, transform, collate_fn, dataloader = None, None, None, None
    else:
        assert isinstance(args.model, str), "model should be a string"
        if args.verbose:
            print(
                f"Load model and use {'no' if args.feature_alignment is None else args.feature_alignment} feature "
                f"alignment",
                flush=True)
        model, transform = load_model(
            model_name=args.model,
            source=args.model_source,
            model_parameters=args.model_parameters,
            module_name=args.module_name,
            feature_alignment=args.feature_alignment,
            device=args.device
        )

        eval_dataset = build_dataset(
            dataset_name=args.dataset,
            root=dataset_root,
            transform=transform,
            split=args.split,  # by default this is the test split
            download=True,
            task=task,
            custom_classname_file=args.custom_classname_file,
            wds_cache_dir=args.wds_cache_dir,
        )
        collate_fn = get_dataset_collate_fn(args.dataset)
        if args.verbose:
            try:
                print(f"Dataset size: {len(eval_dataset)}")
            except TypeError:
                print("IterableDataset has no len()")
            print(f"Dataset split: {args.split}")
            if hasattr(eval_dataset, "classes") and eval_dataset.classes:
                try:
                    print(f"Dataset classes: {eval_dataset.classes}")
                    print(f"Dataset number of classes: {len(eval_dataset.classes)}")
                except AttributeError:
                    print("Dataset has no classes.")

        # Get the dataloader for the split we want to evaluate on, by default this is the test split
        if args.dataset.startswith("wds/"):
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset.batched(args.batch_size), batch_size=None,
                shuffle=False, num_workers=args.num_workers,
            )
        else:
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                collate_fn=collate_fn
            )
        # we also need the train and validation splits for linear probing.
        train_dataset = build_dataset(
            dataset_name=args.dataset,
            root=dataset_root,
            transform=transform,
            split=args.train_split,
            download=True,
        )
        if args.val_split is not None:
            val_dataset = build_dataset(
                dataset_name=args.dataset,
                root=dataset_root,
                transform=transform,
                split=args.val_split,
                download=True,
            )
        elif args.val_proportion is not None:
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,
                                                                       [1 - args.val_proportion,
                                                                        args.val_proportion])
        else:
            val_dataset = None
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            collate_fn=collate_fn, pin_memory=True,
        )
        if val_dataset is not None:
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                collate_fn=collate_fn, pin_memory=True,
            )
        else:
            val_dataloader = None

        return model, train_dataloader, val_dataloader, eval_dataloader


def run(args):
    # device
    args.device = prepare_device(args.distributed)
    # set seed.
    torch.manual_seed(args.seed)
    # fix task
    task = args.task
    mode = args.mode
    # prepare dataset name
    dataset_name = prepare_ds_name(args.dataset)
    # if task == "auto":
    #     task = get_dataset_default_task(dataset_name)

    feature_dirs, model_dirs, out_dir_root, out_dir_pred, model_ids = make_paths(args, dataset_name)

    if dataset_name.startswith('wds'):
        dataset_root = os.path.join(args.dataset_root, 'wds', f'wds_{dataset_name.replace("/", "-")}')
    else:
        dataset_root = args.dataset_root

    if args.verbose:
        print(f"Running '{task}' with mode '{mode}' on '{dataset_name}' with the model(s) '{model_ids}'")

    if task == 'linear_probe':
        if mode == "single_model":
            model, train_dataloader, val_dataloader, eval_dataloader = get_extraction_model_n_dataloader(args,
                                                                                                         dataset_root,
                                                                                                         task)
            evaluator = SingleModelEvaluator(
                model=model, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
                val_dataloader=val_dataloader, normalize=args.normalize, model_id=model_ids[0],
                feature_dir=feature_dirs[0], batch_size=args.batch_size, num_workers=args.num_workers,
                lr=args.fewshot_lr, epochs=args.fewshot_epochs, seed=args.seed, device=args.device,
                fewshot_k=args.fewshot_k, amp=args.amp, probe_out_dir=model_dirs[0], verbose=args.verbose
            )

        elif mode == "combined_models":
            feature_combiner_cls = get_feature_combiner_cls(args.feature_combiner)
            evaluator = CombinedModelEvaluator(
                feature_dirs=feature_dirs, feature_combiner_cls=feature_combiner_cls,
                batch_size=args.batch_size, num_workers=args.num_workers, lr=args.fewshot_lr,
                epochs=args.fewshot_epochs, seed=args.seed, device=args.device, fewshot_k=args.fewshot_k,
                use_val_ds=args.val_proportion is not None or args.val_split is not None,
                normalize=args.normalize, amp=args.amp, probe_out_dir=model_dirs[0], verbose=args.verbose
            )

        elif mode == "ensemble":
            evaluator = EnsembleModelEvaluator(
                model_ids=model_ids, feature_dirs=feature_dirs, linear_prob_dirs=model_dirs,
                batch_size=args.batch_size, num_workers=args.num_workers, lr=args.fewshot_lr,
                epochs=args.fewshot_epochs, seed=args.seed, device=args.device, fewshot_k=args.fewshot_k,
                use_val_ds=args.val_proportion is not None or args.val_split is not None,
                normalize=args.normalize, amp=args.amp, probe_out_dir=out_dir_pred, verbose=args.verbose,
            )

        else:
            raise ValueError(
                "Unsupported mode: {}. mode should be `single_model`, `combined_models`, or `ensemble`".format(
                    mode))
    else:
        raise ValueError(
            "Unsupported task: {}. task should be `linear_probe`".format(
                task))

    metrics = evaluator.evaluate()

    save_results(
        args=args,
        model_ids=model_ids,
        metrics=metrics,
        out_path=out_dir_root
    )
    return 0


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


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
