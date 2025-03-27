import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from helper import format_path
from project_location import FEATURES_ROOT, SUBSET_ROOT, DATASETS_ROOT


def main(args):
    random.seed(args.seed)

    targets_fn = os.path.join(args.imagenet_targets_root, f'targets_{args.split}.pt')
    if not os.path.exists(targets_fn):
        raise FileNotFoundError(f'Targets file not found at {targets_fn}. Please provide the path to the result of the'
                                f' extraction process of any model on ImageNet1k (dataset `wds/imagenet1k`). The '
                                f'targets of the extraction process are the same for all models, so you can use any '
                                f'model\'s ')
    targets = np.array(torch.load(targets_fn))

    unique_classes = np.unique(targets)

    cls_sample_idx_map = {}
    for cls in tqdm(unique_classes):
        indices = np.argwhere(cls == targets).flatten().tolist()
        if args.with_replacement:
            sampled_indices = random.choices(indices, k=args.samples_per_class)
        else:
            sampled_indices = random.sample(indices, args.samples_per_class)
        cls_sample_idx_map[cls.item()] = sampled_indices

    out_root = format_path(args.output_root_dir, args.samples_per_class, args.split)
    if not os.path.exists(out_root):
        os.makedirs(out_root)
        print(f'Created directory {out_root}')
    
    if args.store_seed_in_fn:
        out_fn = os.path.join(out_root, f'imagenet-{args.samples_per_class}k-{args.split}-seed-{args.seed}.json')
    else:
        out_fn = os.path.join(out_root, f'imagenet-{args.samples_per_class}k-{args.split}.json')
    
    with open(out_fn, 'w') as f:
        json.dump(cls_sample_idx_map, f)

    print(f'Successfully sampled indices for each class and saved to {out_fn}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Note we use as default the path to the extracted features and targets of ImageNet1k with any model.
    # The targets are for all models the same, so we can use any model's targets.
    parser.add_argument('--imagenet_targets_root',
                        default=os.path.join(FEATURES_ROOT, 'wds_imagenet1k/dinov2-vit-large-p14'),
                        help='Root directory of the extracted features and targets for ImageNet1k.')
    parser.add_argument('--samples-per-class', default=10, type=int)
    parser.add_argument('--split', default='train', choices=['train', 'test'])
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--store_seed_in_fn', action='store_true',
                        help='Store the seed in the output filename if set, otherwise not.')
    parser.add_argument('--with_replacement', action='store_true',
                        help='Sample with replacement if set, otherwise without replacement.')
    parser.add_argument('--output_root_dir',
                        # default=os.path.join(SUBSET_ROOT, 'imagenet-subset-{num_samples_class}k'),
                        default=os.path.join(DATASETS_ROOT, 'imagenet-subset-{num_samples_class}k-bootstrap'),
                        help='Root directory for the output features and targets.')
    args = parser.parse_args()

    # for i in [1, 5, 10, 20, 30, 40]:
    for i in [30]:
        # for split in ['train', 'test']:
        for split in ['train']:
            for seed in range(500):
                print(f"Run generate_imagenet_subset_indices with {i} samples per class on the {split} split ...")
                args.samples_per_class = i
                args.split = split
                args.seed = seed
                try:
                    main(args)
                except ValueError as e:
                    print(f"Error: {e}")
                    continue
