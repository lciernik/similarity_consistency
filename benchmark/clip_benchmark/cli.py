import argparse
import json
import os
import random
import sqlite3
import sys
from copy import copy
from itertools import product, combinations, islice
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch

from clip_benchmark.argparser import get_parser_args, prepare_args, prepare_combined_args, load_model_configs_args
from clip_benchmark.data import (get_feature_combiner_cls)
from clip_benchmark.data.data_utils import get_extraction_model_n_dataloader
from clip_benchmark.tasks import compute_sim_matrix
from clip_benchmark.tasks.linear_probe_evaluator import (SingleModelEvaluator, CombinedModelEvaluator,
                                                         EnsembleModelEvaluator)
from clip_benchmark.utils.path_maker import PathMaker
from clip_benchmark.utils.utils import (as_list,
                                        get_list_of_datasets,
                                        map_to_probe_dataset,
                                        prepare_ds_name,
                                        world_info_from_env,
                                        set_all_random_seeds)


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

    return list(zip(models, srcs, params, module_names, feature_alignments, model_keys))


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

    # metrics
    def flatten_metrics(curr_metrics):
        new_metrics = {}
        if 'train_metrics' in curr_metrics:
            new_metrics.update({f'train_{k}': v for k, v in curr_metrics['train_metrics'].items()})
        if 'test_metrics' in curr_metrics:
            new_metrics.update({f'test_{k}': v for k, v in curr_metrics['test_metrics'].items()})
        new_metrics.update({k: v for k, v in curr_metrics.items() if not isinstance(v, dict)})
        return new_metrics

    flattened_metrics = flatten_metrics(metrics)
    for key, value in flattened_metrics.items():
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
                print(results_current_run[col])
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
    base = load_model_configs_args(base)

    if base.task == "model_similarity":
        main_model_sim(base)
    else:
        main_eval(base)


def main_model_sim(base):
    base.device = prepare_device(base.distributed)

    # Get list of data to evaluate on
    datasets = get_list_of_datasets(base)

    dataset = datasets[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    dataset_name = prepare_ds_name(dataset)

    train_split = base.train_split

    model_ids = as_list(base.model_key)

    feature_root = os.path.join(base.feature_root, dataset_name)

    # Compute CKA matrix
    sim_matrix, model_ids, method_slug = compute_sim_matrix(sim_method=base.sim_method,
                                                            feature_root=feature_root,
                                                            model_ids=model_ids,
                                                            split=train_split,
                                                            kernel=base.sim_kernel,
                                                            rsa_method=base.rsa_method,
                                                            corr_method=base.corr_method,
                                                            backend='torch',
                                                            unbiased=base.unbiased,
                                                            device=base.device,
                                                            sigma=base.sigma, )
    # Save the similarity matrix
    out_path = os.path.join(base.output_root, dataset_name, method_slug)
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
        if base.verbose:
            print(f'\nCreated path ({out_path}), where results are to be stored ...\n')

    out_res = os.path.join(out_path, f'similarity_matrix.pt')
    if base.verbose:
        print(f"\nDump {base.sim_method.upper()} matrix to: {out_res}\n")
    torch.save(sim_matrix, out_res)
    with open(os.path.join(out_path, f'model_ids.txt'), "w") as file:
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

    if base.verbose:
        print(f"\nModels: {models}")
        print(f"Datasets: {datasets}\n")

    if base.mode != "single_model":
        # TODO: implement different ways how to select the model combinations
        # Check not too many models
        n_models = len(models)
        if n_models > 10:
            raise ValueError(f"Too many models ({n_models}) to combine (max. 10 models). "
                             f"Please select a smaller number of models to combine.")
        model_combinations = []
        for i in range(2, n_models + 1):
            model_combinations += list(combinations(models, i))

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

    for model_info, dataset in runs:

        args = copy(base)
        args = arg_fn(args, model_info)
        args.dataset = dataset

        args.train_split = base.train_split
        args.val_proportion = base.val_proportion  # This should be set for WD tuning!
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


def get_base_evaluator_args(
        args: argparse.Namespace,
        feature_dirs: List[str],
        model_dirs: List[str],
        predictions_dir: str
) -> Dict[str, Any]:
    base_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "lr": args.fewshot_lr,
                   "epochs": args.fewshot_epochs, "seed": args.seed, "device": args.device,
                   "fewshot_k": args.fewshot_k, "feature_dirs": feature_dirs, "model_dirs": model_dirs,
                   "predictions_dir": predictions_dir, "normalize": args.normalize, "amp": args.amp,
                   "verbose": args.verbose, "val_proportion": args.val_proportion}
    return base_kwargs


def run(args):
    # device
    args.device = prepare_device(args.distributed)
    # set seed.
    set_all_random_seeds(args.seed)

    # fix task
    task = args.task
    mode = args.mode
    # prepare dataset name
    dataset_name = prepare_ds_name(args.dataset)
    probe_dataset_name = map_to_probe_dataset(dataset_name)

    path_maker = PathMaker(args, dataset_name, probe_dataset_name)

    dirs = path_maker.make_paths()
    feature_dirs, model_dirs, results_dir, predictions_dir, single_prediction_dirs, model_ids = dirs
    if args.verbose:
        print(f"\n{feature_dirs=}, {model_dirs=}, {results_dir=}, "
              f"{predictions_dir=}, {single_prediction_dirs=}, {model_ids=}\n")

    if dataset_name.startswith("wds"):
        dataset_root = os.path.join(
            args.dataset_root,
            "wds",
            f"wds_{args.dataset.replace('wds/', '', 1).replace('/', '-')}"
        )
    else:
        dataset_root = args.dataset_root

    if args.verbose:
        print(f"\nRunning '{task}' with mode '{mode}' on '{dataset_name}' with the model(s) '{model_ids}'\n")

    base_kwargs = get_base_evaluator_args(args, feature_dirs, model_dirs, predictions_dir)

    if task == 'feature_extraction':
        model, train_dataloader, eval_dataloader = get_extraction_model_n_dataloader(args, dataset_root, task)
        evaluator = SingleModelEvaluator(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            **base_kwargs
        )

        if args.verbose:
            print(f"\nExtracting features for {model_ids} on {dataset_name} and storing them in {feature_dirs} ...\n")

        evaluator.ensure_feature_availability()

    elif task == 'linear_probe':

        if mode == "single_model":
            model, train_dataloader, eval_dataloader = get_extraction_model_n_dataloader(args, dataset_root, task)
            evaluator = SingleModelEvaluator(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                **base_kwargs
            )

        elif mode == "combined_models":
            feature_combiner_cls = get_feature_combiner_cls(args.feature_combiner)
            evaluator = CombinedModelEvaluator(
                feature_combiner_cls=feature_combiner_cls,
                **base_kwargs
            )

        elif mode == "ensemble":
            evaluator = EnsembleModelEvaluator(
                model_ids=model_ids,
                single_prediction_dirs=single_prediction_dirs,
                **base_kwargs
            )

        else:
            raise ValueError(
                "Unsupported mode: {}. mode should be `single_model`, `combined_models`, or `ensemble`".format(
                    mode))

        metrics = evaluator.evaluate()

        save_results(
            args=args,
            model_ids=model_ids,
            metrics=metrics,
            out_path=results_dir
        )

    else:
        raise ValueError(
            "Unsupported task: {}. task should be `feature_extraction` or `linear_probe`".format(
                task))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
