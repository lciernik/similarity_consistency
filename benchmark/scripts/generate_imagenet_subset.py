import argparse
from torchvision.datasets import ImageNet
import numpy as np
import random
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--imagenet-root', default='/home/space/diverse_priors/datasets/imagenet_torch')
parser.add_argument('--samples-per-class', default=10, type=int)
parser.add_argument('--output_path', default='imagenet-10k.json')
args = parser.parse_args()

# ImageNet sorts the classes and filenames, therefore indexing is preserved
# we use the train split
dataset = ImageNet(root=args.imagenet_root, split='train')
targets = np.array(dataset.targets)

unique_classes = np.unique(targets)

cls_sample_idx_map = {}
for cls in tqdm(unique_classes):
    # Get indices of images belonging to the current class
    indices = np.argwhere(cls == targets).flatten().tolist()

    # Sample k indices randomly from class indices
    sampled_indices = random.sample(indices, args.samples_per_class)
    cls_sample_idx_map[cls.item()] = sampled_indices

# Write sampled indices to JSON file
with open(args.output_path, 'w') as f:
    json.dump(cls_sample_idx_map, f)

print(f'Successfully sampled indices for each class and saved to {args.output_path}')
