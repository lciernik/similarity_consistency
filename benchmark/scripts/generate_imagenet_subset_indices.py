import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm


def format_path(path, num_samples_class, split):
    return path.format(
        num_samples_class=num_samples_class,
        split=split
    )


def main(args):
    random.seed(args.seed)

    targets_fn = os.path.join(args.imagenet_targets_root, f'targets_{args.split}.pt')
    if not os.path.exists(targets_fn):
        raise FileNotFoundError(f'Targets file not found at {targets_fn}.')
    targets = np.array(torch.load(targets_fn))

    unique_classes = np.unique(targets)

    cls_sample_idx_map = {}
    for cls in tqdm(unique_classes):
        indices = np.argwhere(cls == targets).flatten().tolist()
        sampled_indices = random.sample(indices, args.samples_per_class)
        cls_sample_idx_map[cls.item()] = sampled_indices

    out_fn = os.path.join(args.output_root_dir, f'imagenet-{args.samples_per_class}k-{args.split}.json')
    with open(out_fn, 'w') as f:
        json.dump(cls_sample_idx_map, f)

    print(f'Successfully sampled indices for each class and saved to {out_fn}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Note we use as default the path to the extracted features and targets of ImageNet1k with any model.
    # The targets are for all models the same, so we can use any model's targets.
    parser.add_argument('--imagenet_targets_root',
                        default='/home/space/diverse_priors/features/wds_imagenet1k/dinov2-vit-large-p14',
                        help='Root directory of the extracted features and targets for ImageNet1k.')
    parser.add_argument('--samples-per-class', default=10, type=int)
    parser.add_argument('--split', default='train', choices=['train', 'test'])
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--output_root_dir',
                        default='/home/space/diverse_priors/datasets/imagenet-subset-{num_samples_class}k',
                        help='Root directory for the output features and targets.')
    args = parser.parse_args()

    main(args)
