import argparse
import json
import os
from itertools import product

from clip_benchmark.utils.path_maker import PathMaker
from helper import load_models, get_hyperparams, parse_datasets

parser = argparse.ArgumentParser()
parser.add_argument('--models_config', type=str, default='./models_config.json')
parser.add_argument('--dataset', type=str, default='wds/imagenet1k')
parser.add_argument('--generate_json', type=str, default='', choices=['', 'features', 'probe', 'predictions'])
args = parser.parse_args()

MODELS_CONFIG = args.models_config

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

    datasets = parse_datasets(args.dataset)

    for dataset in datasets:
        print(f"\n\nChecking features, models and prediction for {dataset=}")
        not_finished_features = {}
        not_finished_probe = {}
        not_finished_pred = {}
        pp_dataset = dataset.replace("/", "_")

        for i, (key, _) in enumerate(models.items()):
            # Using Pathmaker:
            for fewshot_k, fewshot_lr, fewshot_epochs, seed in combs:
              
                # Prepare args
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

                pm = PathMaker(args, pp_dataset)

                # check if features are there
                try:
                    feature_dirs = pm._get_feature_dirs()
                    for feature_dir in feature_dirs:
                        if not os.path.exists(f"{feature_dir}/features_train.pt") and not os.path.exists(
                                f"{feature_dir}/features_test.pt"):
                            not_finished_features[key] = feature_dir
                except FileNotFoundError:
                    print(f"Feature root folder for Model {key} does not exist.")

                # check if models are there
                model_dirs = pm._get_model_dirs()
                for model_dir in model_dirs:
                    if not os.path.exists(f"{model_dir}/model.pkl"):
                        not_finished_probe[key] = model_dir

                # check if predictions are there
                pred_path = pm._get_results_dirs()
                if not os.path.exists(f"{pred_path}/predictions.pkl"):
                    not_finished_pred[key] = pred_path

        print("\nModels without features:")
        print("\n".join(not_finished_features.keys()))

        print("\nModels without probe:")
        print("\n".join(not_finished_probe.keys()))

        print("\nModels without predictions:")
        print("\n".join(not_finished_pred.keys()))

        if args.generate_json == "features" and len(not_finished_features.keys()):
            with open(f"not_finished_features_{pp_dataset}.json", "w") as f:
                json.dump({k: models[k] for k in not_finished_features.keys()}, f)
        elif args.generate_json == "probe" and len(not_finished_probe.keys()):
            with open(f"not_finished_probe_{pp_dataset}.json", "w") as f:
                json.dump({k: models[k] for k in not_finished_probe.keys()}, f)
        elif args.generate_json == "predictions" and len(not_finished_pred.keys()):
            with open(f"not_finished_pred_{pp_dataset}.json", "w") as f:
                json.dump({k: models[k] for k in not_finished_pred.keys()}, f)
