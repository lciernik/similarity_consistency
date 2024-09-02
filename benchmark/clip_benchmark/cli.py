import os
import random
import sys
import traceback
from copy import copy
from itertools import product

import torch

from clip_benchmark.argparser import get_parser_args, prepare_args, prepare_combined_args, load_model_configs_args
from clip_benchmark.data import (get_feature_combiner_cls)
from clip_benchmark.data.builder import get_dataset_class_filter
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
                                        set_all_random_seeds, prepare_device, get_combination, get_list_of_models,
                                        save_results, get_base_evaluator_args, check_existing_results,
                                        check_force_train)


def main():
    parser, base = get_parser_args()
    base = load_model_configs_args(base)

    try:
        if base.task == "model_similarity":
            main_model_sim(base)
        else:
            main_eval(base)
    except Exception as e:
        print(f"An error occurred during the run with models {base.model_key}: \n  {e}")
        traceback.print_exc()

        # Append the args.model_key to the failed_models.txt file
        with open(os.path.join(base.output_root, 'failed_models.txt'), 'a') as f:

            array_job_id = int(os.environ["SLURM_ARRAY_JOB_ID"])
            task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
            f.write(f"{base.model_key} LOGID {array_job_id}_{task_id} \n")
            f.write(f"{str(e)}\n")


def main_model_sim(base):
    base.device = prepare_device(base.distributed)

    # Get list of data to evaluate on
    datasets = get_list_of_datasets(base)

    dataset = datasets[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    dataset_name = prepare_ds_name(dataset)

    train_split = base.train_split

    model_ids = as_list(base.model_key)

    feature_root = os.path.join(base.feature_root, dataset_name)
    subset_root = os.path.join(base.subset_root, dataset_name) if base.use_ds_subset else None

    # Compute CKA matrix
    sim_matrix, model_ids, method_slug = compute_sim_matrix(sim_method=base.sim_method,
                                                            feature_root=feature_root,
                                                            model_ids=model_ids,
                                                            split=train_split,
                                                            subset_root=subset_root,
                                                            kernel=base.sim_kernel,
                                                            rsa_method=base.rsa_method,
                                                            corr_method=base.corr_method,
                                                            backend='torch',
                                                            unbiased=base.unbiased,
                                                            device=base.device,
                                                            sigma=base.sigma,
                                                            max_workers=base.max_workers)
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
    (fewshot_k, fewshot_epochs, rnd_seed, regularization), task_id = get_combination(
        base.fewshot_k,
        base.fewshot_epochs,
        base.seed,
        base.regularization,
    )
    base.fewshot_k = fewshot_k
    base.fewshot_epochs = fewshot_epochs
    base.seed = rnd_seed
    base.regularization = regularization
    base.task_id = task_id

    # Get list of models to evaluate
    models = get_list_of_models(base)

    # Get list of data to evaluate on
    datasets = get_list_of_datasets(base)

    if base.verbose:
        print(f"\nModels: {models}")
        print(f"Datasets: {datasets}\n")

    if base.mode in ["combined_models", "ensemble"]:
        # We combine all provided models and assume selection is done beforehand.
        model_combinations = [models, ]
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

    for model_info, dataset in runs:

        args = copy(base)
        args = arg_fn(args, model_info)
        args.dataset = dataset

        try:
            run(args)
        except Exception as e:
            print(
                f"An error occurred during the run with: "
                f"{model_info} and {dataset}. "
                f"Continuing with the next run.",
                flush=True)
            print(e, flush=True)
            failed_path = os.path.join(base.output_root, 'failed_models.txt')
            with open(failed_path, 'a') as f:
                array_job_id = int(os.environ["SLURM_ARRAY_JOB_ID"])
                task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
                f.write(f"{base.model_key} LOGID {array_job_id}_{task_id} \n")
                f.write(f"{str(e)}\n")


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
    probe_dataset_name = map_to_probe_dataset(dataset_name, verbose=args.verbose)
    args.force_train = check_force_train(dataset_name, args.force_train)

    path_maker = PathMaker(args, dataset_name, probe_dataset_name)

    dirs = path_maker.make_paths()
    feature_dirs, model_dirs, results_dir, single_prediction_dirs, model_ids = dirs
    if args.verbose:
        print(f"\n{feature_dirs=}, {model_dirs=}, {results_dir=}, {single_prediction_dirs=}, {model_ids=}\n")

    if args.skip_existing and check_existing_results(results_dir):
        if args.verbose:
            print(f"Skipping existing results in {results_dir=}")
        return 0

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

    base_kwargs = get_base_evaluator_args(args, feature_dirs, model_dirs, results_dir)

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

        if args.verbose:
            print(f"\nFinished feature extraction for {model_ids} on {dataset_name} ...\n")


    elif task == 'linear_probe':
        base_kwargs["logit_filter"] = get_dataset_class_filter(args.dataset, args.device)

        if mode == "single_model":
            evaluator = SingleModelEvaluator(**base_kwargs)

        elif mode == "combined_models":

            feature_combiner_cls = get_feature_combiner_cls(args.feature_combiner)
            evaluator = CombinedModelEvaluator(feature_combiner_cls=feature_combiner_cls,
                                               **base_kwargs)
        elif mode == "ensemble":
            evaluator = EnsembleModelEvaluator(model_ids=model_ids,
                                               single_prediction_dirs=single_prediction_dirs,
                                               **base_kwargs)
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
