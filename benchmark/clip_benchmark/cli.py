"""Console script for clip_benchmark."""
import argparse
import json
import os
import random
import sqlite3
import sys
from copy import copy
from itertools import product, combinations, islice
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch

from clip_benchmark.argparser import get_parser_args, prepare_args, prepare_combined_args
from clip_benchmark.data import (get_feature_combiner_cls)
from clip_benchmark.data.data_utils import get_extraction_model_n_dataloader
from clip_benchmark.tasks import compute_sim_matrix
from clip_benchmark.tasks.linear_probe import SingleModelEvaluator, CombinedModelEvaluator, EnsembleModelEvaluator
from clip_benchmark.utils.utils import (as_list,
                                        get_list_of_datasets,
                                        get_model_id,
                                        get_train_val_splits,
                                        prepare_ds_name, )


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

    hyperparams_slug = get_hyperparams_name(args)
    model_slug = '__'.join(model_ids)
    if task == "linear_probe" and mode == "combined_models":
        model_slug = model_slug + f"_{args.feature_combiner}"

    # Create list of model checkpoint directories (for the linear probe) for each dataset, model_id, and hyperparameter
    # combination
    if task == "linear_probe" and mode == "combined_models":
        model_dirs = [os.path.join(args.model_root, dataset_slug, model_slug, hyperparams_slug)]
    else:
        model_dirs = [os.path.join(args.model_root, dataset_slug, model_id, hyperparams_slug) for model_id in model_ids]

    # Create output path based on the task, mode, dataset, (combined) model_ids
    out_dir_root = os.path.join(args.output_root, task, mode, dataset_slug, model_slug)
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
    if base.task == "model_similarity":
        main_model_sim(base)
    else:
        main_eval(base)


def main_model_sim(base):
    base.device = prepare_device(base.distributed)

    # Get list of data to evaluate on
    datasets = get_list_of_datasets(base)

    # Get train and val splits
    dataset_info = get_train_val_splits(base.train_split, base.val_proportion, datasets)

    dataset = datasets[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    train_split = dataset_info[dataset]["train_split"]

    # Get list of models to evaluate
    models = get_list_of_models(base)

    # Get model ids
    model_ids = [get_model_id(model[0], model[2]) for model in models]
    model_ids = [(model_id + '-' + dataset).replace('/', '_') for model_id in model_ids]

    # Compute CKA matrix
    sim_matrix, model_ids, method_slug = compute_sim_matrix(base.sim_method,
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

    out_res = os.path.join(base.output_root, f'{method_slug}_similarity_matrix.pt')
    if base.verbose:
        print(f"Dump {base.sim_method.upper()} matrix to: {out_res}")
    torch.save(sim_matrix, out_res)
    with open(os.path.join(base.output_root, f'{method_slug}_model_ids.txt'), "w") as file:
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
    dataset_info = get_train_val_splits(base.train_split, base.val_proportion, datasets)

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
            model_combinations += list(islice(combinations(models, i), 10))

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
        # args.val_split = dataset_info[dataset]["val_split"]  # This is currently not used
        args.val_proportion = dataset_info[dataset]["proportion"]  # This should be set for WD tuning!
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
        base_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "lr": args.fewshot_lr,
                       "epochs": args.fewshot_epochs, "seed": args.seed, "device": args.device,
                       "fewshot_k": args.fewshot_k, "normalize": args.normalize, "amp": args.amp,
                       "probe_out_dir": out_dir_pred, "verbose": args.verbose, "val_proportion": args.val_proportion}

        if mode == "single_model":
            model, train_dataloader, eval_dataloader = get_extraction_model_n_dataloader(args, dataset_root, task)
            evaluator = SingleModelEvaluator(
                model=model, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, model_id=model_ids[0],
                feature_dir=feature_dirs[0], **base_kwargs
            )

        elif mode == "combined_models":
            feature_combiner_cls = get_feature_combiner_cls(args.feature_combiner)
            evaluator = CombinedModelEvaluator(
                feature_dirs=feature_dirs, feature_combiner_cls=feature_combiner_cls, **base_kwargs
            )

        elif mode == "ensemble":
            evaluator = EnsembleModelEvaluator(
                model_ids=model_ids, feature_dirs=feature_dirs, linear_prob_dirs=model_dirs, **base_kwargs
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
