import os

from helper import load_models, get_hyperparams
import argparse
import json
from itertools import product
from clip_benchmark.utils.path_maker import PathMaker

parser = argparse.ArgumentParser()
parser.add_argument('--models_config', type=str, default='./models_config.json')
parser.add_argument('--dataset', type=str, default='wds/imagenet1k')
parser.add_argument('--generate_json', type=str, default='', choices=['', 'features', 'probe', 'predictions'])
args = parser.parse_args()

MODELS_CONFIG = args.models_config
DATASETS = args.dataset

BASE_PROJECT_PATH = "/home/space/diverse_priors"
DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets')
FEATURES_ROOT = os.path.join(BASE_PROJECT_PATH, 'features')
MODELS_ROOT = os.path.join(BASE_PROJECT_PATH, 'models')
OUTPUT_ROOT = os.path.join(BASE_PROJECT_PATH, 'results')

if __name__ == "__main__":
    models, n_models = load_models(MODELS_CONFIG)
    hyper_params, num_jobs = get_hyperparams(num_seeds=5, size='imagenet1k')
    args.task = "linear_probe"
    combs = []
    combs.extend(
        list(
            product(
                hyper_params["fewshot_ks"],
                hyper_params["fewshot_lrs"],
                hyper_params["fewshot_epochs"],
                hyper_params["seeds"]
            )
        )
    )
    not_finished_features = {}
    not_finished_probe = {}
    not_finished_pred = {}
    for i, (key, _) in enumerate(models.items()):
        # Using Pathmaker:
        for fewshot_k, fewshot_lr, fewshot_epochs, seed in combs:

            args.mode = "single_model"
            args.dataset_root = DATASETS_ROOT
            args.feature_root = FEATURES_ROOT
            args.model_root = MODELS_ROOT
            args.output_root = OUTPUT_ROOT
            args.model_key = key
            args.feature_combiner = None
            args.batch_size = 1024
            args.verbose = False
            args.fewshot_k = int(fewshot_k)
            args.fewshot_lr = fewshot_lr
            args.fewshot_epochs = fewshot_epochs
            args.seed = seed

            pm = PathMaker(args, DATASETS.replace("/", "_"))

            try:
                feature_dirs = pm._get_feature_dirs()
                # check if features are there
                for feature_dir in feature_dirs:
                    if not os.path.exists(f"{feature_dir}/features_train.pt"):
                        not_finished_features[key] = feature_dirs
                        # print(f"Features for Model {key} do not exist in {feature_dir}.")
            except FileNotFoundError:
                print(f"Feature root folder for Model {key} does not exist.")
            model_dirs = pm._get_model_dirs()
            # check if models are there
            for model_dir in model_dirs:
                if not os.path.exists(f"{model_dir}/model.pkl"):
                    # print(f"Model for Model {key} does not exist in {model_dir}.")
                    not_finished_probe[key] = model_dirs
            _, pred = pm._get_results_and_predictions_dirs()
            # check if predictions are there
            if not os.path.exists(f"{pred}/predictions.pkl"):
                # print(f"Predictions for Model {key} do not exist in {pred}.")
                not_finished_pred[key] = pred
    print("Models without features:")
    print("\n".join(not_finished_features.keys()))

    print("Models without probe:")
    print("\n".join(not_finished_probe.keys()))

    print("Models without predictions:")
    print("\n".join(not_finished_pred.keys()))

    if args.generate_json == "features" and len(not_finished_features.keys()):
        with open("not_finished_features.json", "w") as f:
            json.dump({k: models[k] for k in not_finished_features.keys()}, f)
    elif args.generate_json == "probe" and len(not_finished_probe.keys()):
        with open("not_finished_probe.json", "w") as f:
            json.dump({k: models[k] for k in not_finished_probe.keys()}, f)
    elif args.generate_json == "predictions" and len(not_finished_pred.keys()):
        with open("not_finished_pred.json", "w") as f:
            json.dump({k: models[k] for k in not_finished_pred.keys()}, f)
