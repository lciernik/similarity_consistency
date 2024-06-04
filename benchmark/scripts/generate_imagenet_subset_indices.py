import argparse
import json
import os
import random

import numpy as np
from torchvision.datasets import ImageNet
from tqdm import tqdm


def main(args):
    random.seed(args.seed)

    dataset = ImageNet(root=args.imagenet_root, split=args.split)
    targets = np.array(dataset.targets)

    unique_classes = np.unique(targets)

    cls_sample_idx_map = {}
    for cls in tqdm(unique_classes):
        indices = np.argwhere(cls == targets).flatten().tolist()
        sampled_indices = random.sample(indices, args.samples_per_class)
        cls_sample_idx_map[cls.item()] = sampled_indices

    out_fn = os.path.join(args.imagenet_root, f'imagenet-{args.samples_per_class}k-{args.split}.json')
    with open(out_fn, 'w') as f:
        json.dump(cls_sample_idx_map, f)

    print(f'Successfully sampled indices for each class and saved to {out_fn}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet-root', default='/home/space/diverse_priors/datasets/imagenet_torch')
    parser.add_argument('--samples-per-class', default=10, type=int)
    parser.add_argument('--split', default='train', choices=['train', 'val'])
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility.')
    args = parser.parse_args()

    main(args)
