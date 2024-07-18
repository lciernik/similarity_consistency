import os
import json
from itertools import product
import numpy as np

def load_models(file_path):
    with open(file_path, "r") as file:
        models = json.load(file)
    return models, len(models)


def count_nr_datasets(datasets_path):
    with open(datasets_path, 'r') as f:
        return len(f.readlines())


def prepare_for_combined_usage(models):
    model_names = []
    sources = []
    model_parameters = []
    module_names = []
    for data in models.values():
        model_names.append(data["model_name"])
        sources.append(data["source"])
        model_parameters.append(data["model_parameters"])
        module_names.append(data["module_name"])
    return model_names, sources, model_parameters, module_names


def get_hyperparams(num_seeds=10, size="extended"):
    if size == "small":
        hyper_params = dict(
            fewshot_lrs=['1e-1', '1e-2'],
            fewshot_ks=['-1'],
            fewshot_epochs=['20'],
            reg_lambda='0',
            regularization=["weight_decay"],
            seeds=[str(num) for num in range(num_seeds)],
        )
    elif size == "imagenet1k":
        hyper_params = dict(
            fewshot_lrs=['1e-1', '1e-2', '1e-3', '1e-4'],
            fewshot_ks=['-1'],
            fewshot_epochs=['20'],
            reg_lambda='1e-2',
            regularization=["weight_decay", "L1"],
            seeds=[str(num) for num in range(num_seeds)],
        )
    else:
        hyper_params = dict(
            fewshot_lrs=['0.1', '0.01'],
            fewshot_ks=['-1', '5', '10', '100'],
            fewshot_epochs=['10', '20', '30'],
            reg_lambda='0',
            regularization=["weight_decay"],
            seeds=[str(num) for num in range(num_seeds)],
        )
    cols_for_array = ['fewshot_ks', 'fewshot_epochs', 'regularization', 'seeds']
    num_jobs = np.prod([len(hyper_params[k]) for k in cols_for_array])
    return hyper_params, num_jobs


def format_path(path, num_samples_class, split):
    return path.format(
        num_samples_class=num_samples_class,
        split=split
    )


def parse_datasets(arg):
    if isinstance(arg, list):
        if len(arg) == 1 and os.path.isfile(arg[0]):
            with open(arg[0], 'r') as f:
                datasets = [line.strip() for line in f if line.strip()]
        else:
            datasets = arg
    else:
        datasets = [arg]
    return datasets
