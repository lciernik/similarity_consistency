"""Console script for clip_benchmark."""
import argparse
import csv
import itertools
import json
import os
import random
import sys
from copy import copy
from itertools import product

import torch

from clip_benchmark.datasets.builder import (build_dataset, dataset_collection,
                                             get_dataset_collate_fn,
                                             get_dataset_collection_from_file,
                                             get_dataset_default_task)
from clip_benchmark.metrics import get_feature_combiner_cls
from clip_benchmark.metrics import linear_probe
from clip_benchmark.models import load_model


def parse_str_to_dict(s):
    try:
        print(s, flush=True)
        parsed_dict = json.loads(s)
        return parsed_dict
    except ValueError as e:
        print(e, flush=True)
        raise argparse.ArgumentTypeError("Input is not a valid JSON dictionary.")


def get_parser_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_eval = subparsers.add_parser('eval', help='Evaluate')
    parser_eval.add_argument('--dataset', type=str, default="cifar10", nargs="+",
                             help="Dataset(s) to use for the benchmark. Can be the name of a dataset, or a collection name ('vtab', 'vtab+', 'imagenet_robustness', 'retrieval') or path of a text file where each line is a dataset name")
    parser_eval.add_argument('--dataset_root', default="root", type=str,
                             help="dataset root folder where the datasets are downloaded. Can be in the form of a template depending on dataset name, e.g., --dataset_root='datasets/{dataset}'. This is useful if you evaluate on multiple datasets.")
    parser_eval.add_argument('--split', type=str, default="test", help="Dataset split to use")
    parser_eval.add_argument('--test_split', dest="split", action='store', type=str, default="test",
                             help="Dataset split to use")
    parser_eval.add_argument('--train_split', type=str, nargs="+", default="train", help="Dataset(s) train split names")
    mutually_exclusive = parser_eval.add_mutually_exclusive_group()
    mutually_exclusive.add_argument('--val_split', default=None, type=str, nargs="+",
                                    help="Dataset(s) validation split names. Mutually exclusive with val_proportion.")
    mutually_exclusive.add_argument('--val_proportion', default=None, type=float, nargs="+",
                                    help="what is the share of the train dataset will be used for validation part, if it doesn't predefined. Mutually exclusive with val_split")

    parser_eval.add_argument('--model', type=str, nargs="+", default=["ViT-B-32-quickgelu"],
                             help="Model architecture to use from OpenCLIP")
    parser_eval.add_argument('--model_source',
                             type=str,
                             nargs="+",
                             default=["ssl"],
                             help="For each model, indicate the source of the model. See thingsvision for more details.")
    parser_eval.add_argument('--model_parameters',
                             nargs="+",
                             type=parse_str_to_dict,
                             metavar="{'KEY1':'VAL1','KEY2':'VAL2',...}",
                             help='A dictionary of key-value pairs')
    parser_eval.add_argument('--module_name', type=str, nargs="+", default=["avgpool"], help="Module name")
    parser_eval.add_argument('--eval_combined', action="store_true",
                             help="Whether the features of the different models should be used in combined fashion.")
    parser_eval.add_argument('--feature_combiner', type=str, default="concat",
                             choices=['concat', 'concat_pca'], help="Feature combiner to use")

    parser_eval.add_argument('--task', type=str, default="auto", choices=["linear_probe"],
                             help="Task to evaluate on. With --task=auto, the task is automatically inferred from the dataset.")
    parser_eval.add_argument('--no_amp', action="store_false", dest="amp", default=True,
                             help="whether to use mixed precision")
    parser_eval.add_argument('--num_workers', default=4, type=int)
    parser_eval.add_argument('--fewshot_k', default=-1, type=int,
                             help="for linear probe, how many shots. -1 = whole dataset.")
    parser_eval.add_argument('--fewshot_epochs', default=10, type=int, help="for linear probe, how many epochs.")
    parser_eval.add_argument('--fewshot_lr', default=0.1, type=float,
                             help="for linear probe, what is the learning rate.")
    parser_eval.add_argument("--skip_load", action="store_true",
                             help="for linear probes, when everything is cached, no need to load model.")
    parser_eval.add_argument("--distributed", action="store_true", help="evaluation in parallel")
    parser_eval.add_argument('--seed', default=0, type=int, help="random seed.")
    parser_eval.add_argument('--batch_size', default=64, type=int)
    parser_eval.add_argument('--normalize', default=True, type=bool, help="features normalization")
    parser_eval.add_argument('--model_cache_dir', default=None, type=str,
                             help="directory to where downloaded models are cached")
    parser_eval.add_argument('--feature_root', default="features", type=str,
                             help="feature root folder where the features are stored.")
    parser_eval.add_argument('--custom_classname_file', default=None, type=str,
                             help="use custom json file with classnames for each dataset, where keys are dataset names and values are list of classnames.")
    parser_eval.add_argument('--custom_template_file', default=None, type=str,
                             help="use custom json file with prompts for each dataset, where keys are dataset names and values are list of prompts. For instance, to use CuPL prompts, use --custom_template_file='cupl_prompts.json'")
    parser_eval.add_argument('--dump_classnames', default=False, action="store_true",
                             help="dump classnames to the results json file.")
    parser_eval.add_argument('--dump_templates', default=False, action="store_true",
                             help="dump templates to the results json file.")

    parser_eval.add_argument('--output', default="result.json", type=str,
                             help="output file where to dump the metrics. Can be in form of a template, e.g., --output='{dataset}_{model}_{task}.json'")
    parser_eval.add_argument('--quiet', dest='verbose', action="store_false", help="suppress verbose messages")
    parser_eval.add_argument('--save_clf', default=None, type=str,
                             help="optionally save the classification layer output by the text tower")
    parser_eval.add_argument('--load_clfs', nargs='+', default=[], type=str,
                             help="optionally load and average mutliple layers output by text towers.")
    parser_eval.add_argument('--skip_existing', default=False, action="store_true",
                             help="whether to skip an evaluation if the output file exists.")
    parser_eval.add_argument('--model_type', default="open_clip", type=str, help="clip model type")
    parser_eval.add_argument('--wds_cache_dir', default=None, type=str,
                             help="optional cache directory for webdataset only")
    parser_eval.set_defaults(which='eval')

    parser_build = subparsers.add_parser('build', help='Build CSV from evaluations')
    parser_build.add_argument('files', type=str, nargs="+", help="path(s) of JSON result files")
    parser_build.add_argument('--output', type=str, default="benchmark.csv", help="CSV output file")
    parser_build.set_defaults(which='build')

    args = parser.parse_args()
    return parser, args


def main():
    parser, base = get_parser_args()
    if not hasattr(base, "which"):
        parser.print_help()
        return
    if base.which == "eval":
        main_eval(base)
    elif base.which == "build":
        main_build(base)


def main_build(base):
    # Build a benchmark single CSV file from a set of evaluations (JSON files)
    rows = []
    fieldnames = set()

    def process_file(path: str):
        data = json.load(open(path))
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
    with open(base.output, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def prepare_args(args, model_info):
    args.model = model_info[0]  # model
    args.model_source = model_info[1]  # model_source
    args.model_parameters = model_info[2]  # model_parameters
    args.module_name = model_info[3]  # module_name
    return args


def prepare_combined_args(args, model_comb):
    args.model = [tup[0] for tup in model_comb]
    args.model_source = [tup[1] for tup in model_comb]
    args.model_parameters = [tup[2] for tup in model_comb]
    args.module_name = [tup[3] for tup in model_comb]
    return args


def main_eval(base):
    # Get list of models to evaluate
    models = _as_list(base.model)
    srcs = _as_list(base.model_source)
    params = _as_list(base.model_parameters)
    module_names = _as_list(base.module_name)

    assert len(models) == len(srcs), "The number of model_source should be the same as the number of models"
    assert len(models) == len(params), "The number of model_parameters should be the same as the number of models"
    assert len(models) == len(module_names), "The number of module_name should be the same as the number of models"

    models = list(zip(models, srcs, params, module_names))

    # Get list of datasets to evaluate on
    datasets = []
    for name in _as_list(base.dataset):
        if os.path.isfile(name):
            # If path, read file, each line is a dataset name
            datasets.extend(get_dataset_collection_from_file(name))
        elif name in dataset_collection:
            # if part of `dataset_collection`, retrieve from it
            datasets.extend(dataset_collection[name])
        else:
            # if not, assume it is simply the name of the dataset
            datasets.append(name)

    train_splits = _as_list(base.train_split)
    train_splits = _single_option_to_multiple_datasets(train_splits, datasets, "train_split")
    proportions, val_splits = None, None
    if base.val_split is not None:
        val_splits = _as_list(base.val_split)
        val_splits = _single_option_to_multiple_datasets(val_splits, datasets, "val_split")
    if base.val_proportion is not None:
        proportions = _as_list(base.val_proportion)
        proportions = _single_option_to_multiple_datasets(proportions, datasets, "val_proportion")

    dataset_info = {}
    for i in range(len(datasets)):
        dataset_info[datasets[i]] = {
            "train_split": train_splits[i],
            "val_split": val_splits[i] if val_splits is not None else None,
            "proportion": proportions[i] if proportions is not None else None
        }

    if base.verbose:
        print(f"Models: {models}")
        print(f"Datasets: {datasets}")

    if base.eval_combined:
        # TODO: implement different ways how to select the model combinations
        # Now assumption that passed models are combined together
        n_models = len(models)
        model_combinations = []
        for i in range(2, n_models + 1):
            model_combinations += list(itertools.combinations(models, i))
        
        runs = product(model_combinations, datasets)
        # runs = product([models], datasets)
        arg_fn = prepare_combined_args
        run_fn = run_combined
    else:
        runs = product(models, datasets)
        arg_fn = prepare_args
        run_fn = run

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
        args.train_split = dataset_info[dataset]["train_split"]
        args.val_split = dataset_info[dataset]["val_split"]
        args.val_proportion = dataset_info[dataset]["proportion"]

        try:
            run_fn(args)
        except Exception as e:
            print(
                f"An error occurred during the run with: "
                f"{model_info} and {dataset}. "
                f"Continuing with the next run.",
                flush=True)
            print(e, flush=True)
            raise e


def _as_list(l):
    if not l:
        return []
    return [l] if type(l) != list else l


def _single_option_to_multiple_datasets(cur_option, datasets, name):
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


def prepare_device(distributed):
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


def prepare_ds_name(dataset):
    if dataset.startswith("wds/"):
        return dataset.replace("wds/", "", 1)
    else:
        return dataset


def make_output_fname(args, dataset_name, task):
    # dataset name for outpath
    dataset_slug = dataset_name.replace('/', '_')

    # model name for outpath and model ids
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

    if isinstance(args.model, list):
        model_ids = [_get_model_id(model, model_params) for model, model_params in
                     zip(args.model, args.model_parameters)]
        model_slug = '__'.join(model_ids)
        if args.eval_combined:
            model_slug += f"_feat_comb_{args.feature_combiner}"
    else:
        model_slug = _get_model_id(args.model, args.model_parameters)
        model_ids = [model_slug]

    fewshot_slug = "no_fewshot" if args.fewshot_k == -1 else f"fewshot_{args.fewshot_k}"
    # output file name
    output = args.output.format(
        model=model_slug,
        task=task,
        dataset=dataset_slug,
        fewshot_k=fewshot_slug,
        seed=args.seed
    )
    return output, model_ids


def run_combined(args):
    # device
    args.device = prepare_device(args.distributed)
    # set seed.
    torch.manual_seed(args.seed)
    # fix task
    task = args.task
    # prepare dataset name
    dataset_name = prepare_ds_name(args.dataset)
    if task == "auto":
        task = get_dataset_default_task(dataset_name)

    output, model_ids = make_output_fname(args, dataset_name, task)

    if args.verbose:
        print(f"\n .... Running '{task}' on '{dataset_name}' with the combined models '{'__'.join(model_ids)}' ....\n")

    if os.path.exists(output) and args.skip_existing:
        if args.verbose:
            print(f"Skip {output}, exists already.")
        return

    if task == "linear_probe":

        model_ids_w_ds = [(model_id + '-' + args.dataset).replace('/', '_') for model_id in model_ids]

        ## get feature combiner cls
        feature_combiner_cls = get_feature_combiner_cls(args.feature_combiner)

        metrics = linear_probe.evaluate_combined(
            model_ids=model_ids_w_ds,
            feature_combiner_cls=feature_combiner_cls,
            feature_root=args.feature_root,
            fewshot_k=args.fewshot_k,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.fewshot_lr,
            epochs=args.fewshot_epochs,
            device=args.device,
            seed=args.seed,
            use_val_ds=args.val_proportion is not None or args.val_split is not None,
            amp=args.amp,
            verbose=args.verbose
        )
    else:
        raise ValueError(
            "Unsupported task: {}. task should be `zeroshot_classification`, `zeroshot_retrieval`, `linear_probe`, or `captioning`".format(
                task))

    dump = {
        "dataset": args.dataset,
        "model_ids": model_ids,
        "model": args.model,
        "model_source": args.model_source,
        "model_parameters": args.model_parameters,
        "module_name": args.module_name,
        "task": task,
        "metrics": metrics,
    }
    # if hasattr(dataset, "classes") and dataset.classes and args.dump_classnames:
    #     dump["classnames"] = dataset.classes
    # if hasattr(dataset, "templates") and dataset.templates and args.dump_templates:
    #     dump["templates"] = dataset.templates
    if args.verbose:
        print(f"Dump results to: {output}")
    with open(output, "w") as f:
        json.dump(dump, f)
    return 0


def run(args):
    """Console script for clip_benchmark."""
    # device 
    args.device = prepare_device(args.distributed)
    # set seed.
    torch.manual_seed(args.seed)
    # fix task
    task = args.task
    # prepare dataset name
    dataset_name = prepare_ds_name(args.dataset)
    if task == "auto":
        task = get_dataset_default_task(dataset_name)

    output, model_ids = make_output_fname(args, dataset_name, task)
    model_id = model_ids[0]

    if os.path.exists(output) and args.skip_existing:
        if args.verbose:
            print(f"Skip {output}, exists already.")
        return

    if args.verbose:
        print(f"Running '{task}' on '{dataset_name}' with the model '{model_id}'")

    dataset_root = args.dataset_root.format(dataset=dataset_name, dataset_cleaned=dataset_name.replace("/", "-"))

    if args.skip_load or isinstance(args.model, list):
        model, transform, collate_fn, dataloader = None, None, None, None
    else:
        assert isinstance(args.model, str), "model should be a string"
        model, transform = load_model(
            model_name=args.model,
            source=args.model_source,
            model_parameters=args.model_parameters,
            module_name=args.module_name,
            device=args.device
        )
        dataset = build_dataset(
            dataset_name=args.dataset,
            root=dataset_root,
            transform=transform,
            split=args.split,
            download=True,
            task=task,
            custom_template_file=args.custom_template_file,
            custom_classname_file=args.custom_classname_file,
            wds_cache_dir=args.wds_cache_dir,
        )
        collate_fn = get_dataset_collate_fn(args.dataset)
        if args.verbose:
            try:
                print(f"Dataset size: {len(dataset)}")
            except TypeError:
                print("IterableDataset has no len()")
            print(f"Dataset split: {args.split}")
            if hasattr(dataset, "classes") and dataset.classes:
                try:
                    print(f"Dataset classes: {dataset.classes}")
                    print(f"Dataset number of classes: {len(dataset.classes)}")
                except AttributeError:
                    print("Dataset has no classes.")

        if args.dataset.startswith("wds/"):
            dataloader = torch.utils.data.DataLoader(
                dataset.batched(args.batch_size), batch_size=None,
                shuffle=False, num_workers=args.num_workers,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                collate_fn=collate_fn
            )

    if task == "linear_probe":
        # we also need the train and validation splits for linear probing.
        train_dataset = None
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
                                                                       [1 - args.val_proportion, args.val_proportion])
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
        metrics = linear_probe.evaluate(
            model,
            train_dataloader,
            dataloader,
            args.fewshot_k,
            args.batch_size,
            args.num_workers,
            args.fewshot_lr,
            args.fewshot_epochs,
            (model_id + '-' + args.dataset).replace('/', '_'),
            args.seed,
            args.feature_root,
            val_dataloader=val_dataloader,
            device=args.device,
            normalize=args.normalize,
            amp=args.amp,
            verbose=args.verbose,
        )
    else:
        raise ValueError(
            "Unsupported task: {}. task should be `zeroshot_classification`, `zeroshot_retrieval`, `linear_probe`, or `captioning`".format(
                task))
    dump = {
        "dataset": args.dataset,
        "model_ids": model_ids,
        "model": args.model,
        "model_source": args.model_source,
        "model_parameters": args.model_parameters,
        "module_name": args.module_name,
        "task": task,
        "metrics": metrics,
    }
    if hasattr(dataset, "classes") and dataset.classes and args.dump_classnames:
        dump["classnames"] = dataset.classes
    if hasattr(dataset, "templates") and dataset.templates and args.dump_templates:
        dump["templates"] = dataset.templates
    if args.verbose:
        print(f"Dump results to: {output}")
    with open(output, "w") as f:
        json.dump(dump, f)
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
