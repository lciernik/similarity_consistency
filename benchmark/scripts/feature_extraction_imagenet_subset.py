"""
Extract ImageNet features for a subset of samples per class.
Subsets are taken by already extracted features and targets of ImageNet1k.
"""
import argparse
import json
import os

import numpy as np
import torch

from clip_benchmark.utils.utils import load_features_targets


def format_path(path, num_samples_class, split):
    return path.format(
        num_samples_class=num_samples_class,
        split=split
    )


def main(args):
    # Create output directory: replace {num_samples_class} and {split} in the path
    out_path_root = format_path(args.output_root_dir, args.num_samples_class, args.split)

    # Get path to the subset indices file: replace {num_samples_class} and {split} in the path
    idxs_fn = format_path(args.subset_idxs, args.num_samples_class, args.split)
    # Load the indices map
    with open(idxs_fn, 'r') as f:
        indices_map = json.load(f)
    indices = np.array(list(map(list, indices_map.values()))).flatten()

    model_keys = [args.model_key] if not isinstance(args.model_key, list) else args.model_key

    for model_id in model_keys:
        try:
            features, targets = load_features_targets(args.features_root, model_id, args.split)
        except FileNotFoundError as e:
            print(f'Features of ImageNet1k not found for model {model_id}. Skipping...')
            print(f'>> Error: {e}')
            continue

        features_subset = features[indices, :]
        targets_subset = targets[indices]

        feature_dir = os.path.join(out_path_root, model_id)
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
            print(f'Created directory {feature_dir}')

        torch.save(features_subset, os.path.join(feature_dir, f'features_{args.split}.pt'))
        torch.save(targets_subset, os.path.join(feature_dir, f'targets_{args.split}.pt'))
        print(f'Saved features and targets for model {model_id} to {feature_dir}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_root', default='/home/space/diverse_priors/features/wds_imagenet1k',
                        help='Root directory of the extracted features and targets for ImageNet1k.')
    parser.add_argument('--model_key', nargs='+', default=['dinov2-vit-large-p14'],
                        help='Model key(s) for which the features are extracted.')
    parser.add_argument('--split', default='train', choices=['train', 'test'])
    parser.add_argument('--num_samples_class', default=10, type=int,
                        help='Number of samples per class in the subset.')
    parser.add_argument('--subset_idxs',
                        default='/home/space/diverse_priors/datasets/imagenet_torch/imagenet-{num_samples_class}k-{split}.json',
                        help='Path to the subset indices file.')
    parser.add_argument('--output_root_dir',
                        default='/home/space/diverse_priors/features/imagenet-subset-{num_samples_class}k',
                        help='Root directory for the output features and targets.')
    args = parser.parse_args()

    main(args)
