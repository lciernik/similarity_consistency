import os
import json
import sqlite3
import pandas as pd
import argparse
from typing import Dict, List, Optional

from helper import load_models, get_hyperparams
from clip_benchmark.utils.utils import prepare_ds_name
from clip_benchmark.utils.path_maker import PathMaker

parser = argparse.ArgumentParser()
parser.add_argument('--models_config', type=str, default='./models_config.json')
parser.add_argument('--datasets', type=str, nargs="+",
                    default=["wds_imagenet-a","wds_imagenet-r","wds_imagenet_sketch","wds_imagenetv2", 'wds_imagenet1k'])
parser.add_argument('--mode', type=str,
                    choices=["single_model", "ensemble", "combined_models"],
                    default='single_model')
parser.add_argument('--hyperparams', type=str, default='imagenet1k')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--verbose', type=bool, default=False)

args = parser.parse_args()

BASE_PROJECT_PATH = "/home/space/diverse_priors"
SAMPLING_ROOT = os.path.join(BASE_PROJECT_PATH, 'sampling')
RESULTS_ROOT = os.path.join(BASE_PROJECT_PATH, 'results')
OUTPUT_ROOT = os.path.join(RESULTS_ROOT, "aggregated")
# Needed for PathMaker only
DATASETS_ROOT = os.path.join(BASE_PROJECT_PATH, 'datasets')
FEATURES_ROOT = os.path.join(BASE_PROJECT_PATH, 'features')
MODELS_ROOT = os.path.join(BASE_PROJECT_PATH, 'models')


def retrieve_db_results(db_path:str, verbose:bool) -> pd.DataFrame:
    path = db_path
    try:
        conn = sqlite3.connect(path)
        df = pd.read_sql('SELECT * FROM "results"', conn)
        conn.close()
    except (pd.errors.DatabaseError, sqlite3.OperationalError) as e:
        if verbose:
            print(f"ERROR: Tried to extract data from {path=}, but got Error: {e}")
            print(f"{os.path.exists(path)=}")
        df = None
    return df


def get_processed_hyperparams(size: str, batch_size: int) -> Dict[str, List]:
    hyper_params, _ = get_hyperparams(num_seeds=1, size=size)
    hyper_params["fewshot_lr"] = hyper_params.pop("fewshot_lrs")
    hyper_params["fewshot_k"] = hyper_params.pop("fewshot_ks")
    del hyper_params["seeds"]
    hyper_params["batch_size"] = [batch_size]

    for k,v in hyper_params.items():
        try:
            hyper_params[k] = [float(x) for x in v]
        except ValueError:
            pass

    return hyper_params


def save_dataframe(out_df: pd.DataFrame, dataset:str, mode:str, hyperparams:str, verbose:bool=False):
    if out_df.empty:
        print(f"Empty dataframe for {dataset}. Skipping.")
    else:
        out_path = os.path.join(OUTPUT_ROOT, dataset, mode)
        file_path = os.path.join(out_path, f"results_{hyperparams}.pkl")
        os.makedirs(out_path, exist_ok=True)
        if verbose:
            print(f"Saving {len(out_df)} rows to", file_path)
        out_df.to_pickle(file_path)


def resolve_directory_name(dir_name: str) -> Dict:
    info = {}
    param_names_map = {"models": "n_models", "samples": "n_samples"}
    dir_name_parts = dir_name.split(os.sep)[-1]
    for part in dir_name_parts.split("-"):
        k, v = part.split("_")
        if k in param_names_map:
            info[param_names_map[k]] = int(v)
        else:
            raise ValueError(f"Unknown directory name component {k} of directory {dir_name}")
    return info


def resolve_similarity_slug(similarity_slug: Optional[str]) -> Dict:
    keys = ["similarity_method", "cka_kernel", "cka_sigma", "cka_unbiased", "rsa_method", "rsa_correlation"]
    info = {k: None for k in keys}
    if similarity_slug is not None:
        parts = similarity_slug.split("_")
        info["similarity_method"] = parts.pop(0)
        if info["similarity_method"] == "cka":
            info["cka_unbiased"] = False

        skip = 0
        for p_i, part in enumerate(parts):
            if skip > 0:
                skip -= 1
                continue
            if part == "kernel":
                info["cka_kernel"] = parts[p_i + 1]
                skip = 1
            elif part == "sigma":
                info["cka_sigma"] = float(parts[p_i + 1])
                skip = 1
            elif part == "unbiased":
                info["cka_unbiased"] = True
                skip = 1
            elif part == "method":
                info["rsa_method"] = parts[p_i + 1]
                skip = 1
            elif part == "corr":
                info["rsa_correlation"] = parts[p_i + 2]
                skip = 2
    return info


def resolve_file_name(file_name: str) -> Dict:
    info = {}
    file_name = file_name.replace(".json", "")
    param_names = ["sampling_method", "similarity_method", "num_clusters", "clustering_method", "cluster_id"]

    if "_" not in file_name:
        # Parameterless samplings
        info["sampling_method"] = file_name
        for pn in param_names:
            if pn not in info:
                info[pn] = None
    else:
        # Samplings with parameters
        file_parts = file_name.split("-")
        for param_name, part in zip(param_names, file_parts):
            if param_name in ["sampling_method", "similarity_method", "clustering_method"]:
                info[param_name] = part
            else:
                part = part.split("_")[-1]
                try:
                    info[param_name] = int(part)
                except ValueError:
                    info[param_name] = part

    info.update(resolve_similarity_slug(info["similarity_method"]))

    return info


def load_sampling_info(sampling_root: str) -> List[Dict]:
    info = []
    for root, dirs, files in os.walk(sampling_root):
        for file in files:
            if file.endswith(".json"):
                sampling_dict = resolve_directory_name(root)
                sampling_dict.update(resolve_file_name(file))

                # Add the names of the sampled models, for each sample
                with open(os.path.join(root, file), 'r') as f:
                    list_of_samples = json.load(f)
                for i, model_sample in enumerate(list_of_samples):
                    one_sample_dict = sampling_dict.copy()
                    one_sample_dict["models"] = sorted(model_sample)
                    one_sample_dict["sample_id"] = i
                    info.append(one_sample_dict)
    return info


def build_dataframe_for_dataset(dataset: str, models: List, hyper_params: Dict, args) -> pd.DataFrame:
    dataset = prepare_ds_name(dataset)
    out_df = pd.DataFrame()

    for model_id in models:
        args.model_key = model_id
        pm = PathMaker(args, dataset)
        pm.make_paths()

        db_path = os.path.join(pm._get_results_and_predictions_dirs()[0], "results.db")

        if not os.path.exists(db_path):
            print(f"Database not found for {dataset} and {model_id}. Skipping.")
            print(f" Location", db_path)
            continue

        df = retrieve_db_results(db_path, verbose=args.verbose)
        if df is None:
            print(f"Error loading database for {dataset} and {model_id}. Skipping.")
            continue

        # Filter by hyperparams
        for k, v in hyper_params.items():
            df = df[df[k].isin(v)]

        if df.empty:
            print(f"Empty dataframe for {dataset} and {model_id}. Skipping.")
        else:
            if args.verbose:
                print(f"Found {len(df)} entries for {dataset} and {model_id}.")
            out_df = pd.concat([out_df, df])

        # Remove clutter
        for c in ["train_split", "test_split", "module_name", "model_parameters"]:
            if c in out_df.columns:
                del out_df[c]
    return out_df


if __name__ == "__main__":
    args.dataset_root = DATASETS_ROOT
    args.feature_root = FEATURES_ROOT
    args.model_root = MODELS_ROOT
    args.output_root = RESULTS_ROOT
    args.feature_combiner = None
    args.seed = 0
    args.task = "linear_probe"
    hyper_params = get_processed_hyperparams(args.hyperparams, args.batch_size)
    # We only need hyperparams in args to instantiate the PathMaker. We pick "real" hyperparams to avoid creating new folders.
    args.fewshot_k = hyper_params["fewshot_k"][0]
    args.fewshot_lr = hyper_params["fewshot_lr"][0]
    args.fewshot_epochs = hyper_params["fewshot_epochs"][0]

    if args.mode == "single_model":
        models, _ = load_models(args.models_config)
        for dataset in args.datasets:
            out_df = build_dataframe_for_dataset(dataset, models.keys(), hyper_params, args)
            del out_df["combiner"]
            del out_df["model"]
            save_dataframe(out_df, dataset, args.mode, args.hyperparams, verbose=args.verbose)

    elif args.mode in ("ensemble", "combined_models"):
        sampling_info = load_sampling_info(SAMPLING_ROOT)
        out_dfs = []
        for dataset in args.datasets:
            for one_sample_info in sampling_info:
                models = [one_sample_info["models"]]

                df = build_dataframe_for_dataset(dataset, models, hyper_params, args)

                info_df = pd.DataFrame([one_sample_info] * len(df))
                df = pd.concat([df, info_df], axis=1)

                out_dfs.append(df)
            out_df = pd.concat(out_dfs, ignore_index=True)
            del out_df["model"]
            save_dataframe(out_df, dataset, args.mode, args.hyperparams, verbose=args.verbose)

    if args.verbose:
        print("Done.")
