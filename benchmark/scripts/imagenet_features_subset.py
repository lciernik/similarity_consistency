import argparse
import json
import os
import random

import numpy as np
from torchvision.datasets import ImageNet
from tqdm import tqdm

def format_path(path, num_samples_class, split):
    return path.format(
        num_samples_class=num_samples_class,
        split=split
    ) 
def main(args):
    out_path_root = format_path(args.output_root_dir, args.num_samples_class, args.split)

    idxs_fn = format_path(args.subset_idxs, args.num_samples_class, args.split)
    indices_map = json.load(idxs_fn)
    indices = np.array(list(map(list, indices_map.values()))).flatten()
    indices = np.sort(indices)
    
    model_keys = as_list(args.model_key)
    for model_id in model_keys:
        features, targets = load_features_targets(args.features_root, model_id, args.split) 
        features_subset = features[indices,:]
        targets_subset = targets[indices,:]
        #TODO storing at the correct position


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_root', default='/home/space/diverse_priors/features/wds_imagenet1k')
    parser.add_argument('--model_key', nargs='+')
    parser.add_argument('--split', default='train')
    parser.add_argument('--num_samples_class', default=10, type=int)
    parser.add_argument('--subset_idxs', default='/home/space/diverse_priors/datasets/imagenet_torch/imagenet-{num_samples_class}k-{split}.json')
    parser.add_argument('--output_root_dir', default='/home/space/diverse_priors/features/imagenet-subset-{num_samples_class}k')
    args = parser.parse_args()

    main(args)