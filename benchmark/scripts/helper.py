import json
from itertools import product

def load_models(file_path):
    with open(file_path, "r") as file:
        models = json.load(file)
    return models, len(models)


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


def get_hyperparams(num_seeds=10):
    hyper_params = dict(
        fewshot_lrs=['0.1', '0.01'],
        fewshot_ks=['-1', '5', '10', '100'],
        fewshot_epochs=['10', '20', '30'],
        seeds=[str(num) for num in range(num_seeds)],
    )
    num_jobs = len(list(product(*hyper_params.values())))
    return hyper_params, num_jobs