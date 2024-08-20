import argparse
import json

import torch

from clip_benchmark.models import load_model
from helper import load_models


def format_params(num_params):
    if num_params >= 1_000_000_000:  # Billions
        return f'{num_params / 1_000_000_000:.1f}B'
    elif num_params >= 1_000_000:  # Millions
        return f'{num_params / 1_000_000:.1f}M'
    elif num_params >= 1_000:  # Thousands
        return f'{num_params / 1_000:.1f}K'
    else:  # Less than a thousand
        return str(num_params)


def class_params(num_params):
    if num_params < 100_000_000:  # < 100M
        return 'small'
    elif num_params < 300_000_000:  # < 300M
        return 'medium'
    elif num_params < 400_000_000:  # < 400M
        return 'large'
    else:
        return 'xlarge'


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_config, nmodels = load_models(args.models_config)

    new_models_config = {}

    for model_id, config in models_config.items():
        if args.verbose:
            print(f"\nRetrieving number of parameters for {model_id=}\n")
        model, _ = load_model(
            config['source'],
            config['model_name'],
            config['module_name'],
            config['model_parameters'],
            config['alignment'],
            device
        )
        model_size = model.n_parameters()
        config['size'] = model_size
        config['size_fmt'] = format_params(model_size)
        config['size_class'] = class_params(model_size)
        new_models_config[model_id] = config

    with open(args.out_fn, 'w') as f:
        json.dump(new_models_config, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_config', type=str, default='./models_config.json')
    parser.add_argument('--out_fn', type=str, default='./models_config.json')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(args)
