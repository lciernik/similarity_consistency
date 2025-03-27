"""
Extract ImageNet features for a subset of samples per class.
Subsets are taken by already extracted features and targets of ImageNet1k.
"""
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from sim_consistency.utils.utils import load_features_targets
from helper import load_models, format_path
from project_location import FEATURES_ROOT, SUBSET_ROOT

MODELS_CONFIG = "./configs/models_config_wo_alignment.json"

def new_format_path(path, num_samples_class, split, seed):
    return path.format(num_samples_class=num_samples_class, split=split, seed=seed)

def main(args):
    # Get the model keys
    model_keys = [args.model_key] if not isinstance(args.model_key, list) else args.model_key

    print(f"For each model in model_keys (n={len(model_keys)}) extract the features ...")
    for model_id in model_keys:
        print(f"\n\n> Model: {model_id}\n")

        try:
            features, targets = load_features_targets(args.features_root, model_id, args.split)
        except FileNotFoundError as e:
            print(
                f'\nFeatures or targets of wds_imagenet1k not found for model {model_id} and idxes at:\n{idxs_fn}. Skipping...')
            print(f'>> Error: {e}\n')
            continue

        # Iterate over the seeds and extract the features
        for seed in range(args.max_seed):
            print(f"\n>> Seed: {seed}\n")

            # Create output directory: replace {num_samples_class} and {split} in the path
            out_path_root = new_format_path(args.output_root_dir, args.num_samples_class, args.split, seed)
            print(f">> Formatted {out_path_root=}")

            # Get path to the subset indices file: replace {num_samples_class} and {split} in the path
            idxs_fn = new_format_path(args.subset_idxs, args.num_samples_class, args.split, seed)
            
            # Load the indices map
            with open(idxs_fn, 'r') as f:
                indices_map = json.load(f)
            indices = np.array(list(map(list, indices_map.values()))).flatten()
            print(f">> Loaded indices from {idxs_fn=}")

            feature_dir = os.path.join(out_path_root, model_id)
            feat_fn = os.path.join(feature_dir, f'features_{args.split}.pt')
            tar_fn = os.path.join(feature_dir, f'targets_{args.split}.pt')

            if os.path.exists(feat_fn) and os.path.exists(tar_fn):
                print(f">> Features and targets for model {model_id} already exist at {feature_dir}. Skipping...")
                continue

            features_subset = features[indices, :]
            targets_subset = targets[indices]
            
            if not os.path.exists(feature_dir):
                os.makedirs(feature_dir)
                print(f'Created directory {feature_dir}')

            torch.save(features_subset, feat_fn)
            torch.save(targets_subset, tar_fn)
            print(f'>> Saved {args.split} features and targets for model {model_id} to:\n{feature_dir}\n')


if __name__ == "__main__":
    models, n_models = load_models(MODELS_CONFIG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--features_root', default=os.path.join(FEATURES_ROOT, 'wds_imagenet1k'),
                        help='Root directory of the extracted features and targets for ImageNet1k.')
    parser.add_argument('--model_key', nargs='+', default=list(models.keys()),
                        help='Model key(s) for which the features are extracted.')
    parser.add_argument('--split', default='train', choices=['train', 'test'])
    parser.add_argument('--num_samples_class', default=10, type=int,
                        help='Number of samples per class in the subset.')
    parser.add_argument('--subset_idxs',
                        required=True,
                        help='Path to the subset indices file.')
    parser.add_argument('--output_root_dir',
                        required=True,
                        help='Root directory for the output features and targets.')
    parser.add_argument('--max_seed', default=3, type=int,
                        help='Maximum seed for the subset indices.')
    args = parser.parse_args()

    main(args)
