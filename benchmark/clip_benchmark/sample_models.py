import os.path
from typing import Dict, Callable
import json
from enum import Enum
from typing import List
from analysis.utils import retrieve_performance
from analysis.samplers import TopKSampler, RandomSampler
from functools import partial


class SamplingStrategy(Enum):
    TOP_K = 'top-k'
    RANDOM = 'random'
    CLUSTER = 'cluster'
    ONE_CLUSTER = 'one_cluster'


def build_sampler(sampling_strategy: SamplingStrategy, num_models: int,
                  models: Dict, selection_dataset: str):
    # Returns the performance for a given model
    model_scoring_fn = partial(retrieve_performance, dataset_id=selection_dataset)

    default_args = dict(k=num_models, models=models)
    if sampling_strategy == SamplingStrategy.RANDOM:
        sampler = RandomSampler(**default_args)
    elif sampling_strategy == SamplingStrategy.TOP_K:
        sampler = TopKSampler(**default_args, model_scoring_fn=model_scoring_fn)
    else:
        raise ValueError(f"Unknown sampling strategy. Possible values are {list(SamplingStrategy)}")
    return sampler


def main(num_models: int,
         sampling_strategies: List[SamplingStrategy],
         output_root: str,
         model_config_path: str,
         selection_dataset: str,
         num_samples: int):
    # Retrieve model definitions
    with open(model_config_path, 'r') as f:
        models = json.load(f)

    os.makedirs(output_root, exist_ok=True)

    for sampling_strategy in sampling_strategies:
        sampler = build_sampler(sampling_strategy=SamplingStrategy(sampling_strategy),
                                num_models=num_models,
                                models=models,
                                selection_dataset=selection_dataset)

        model_sets = []
        n_sets = min(num_samples, sampler.max_available_samples())
        for _ in range(n_sets):
            model_set = sampler.sample()
            model_sets.append(model_set)

        output_file = os.path.join(output_root, f'{sampling_strategy}.json')
        with open(output_file, 'w') as f:
            json.dump(model_sets, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_models', type=int)
    parser.add_argument('--sampling_strategies', nargs='+', choices=[s.value for s in SamplingStrategy])
    parser.add_argument('--model_config_path', default='scripts/models_config.json')
    parser.add_argument('--selection_dataset', default='imagenet-1k')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_root')
    args = parser.parse_args()

    main(**vars(args))
