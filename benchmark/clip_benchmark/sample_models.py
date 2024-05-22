import os.path
from typing import Dict
import random
import json
from enum import Enum
from typing import List
from sampling.utils import retrieve_imagenet_performance


class SamplingStrategy(Enum):
    TOP_K = 'top-k'
    RANDOM = 'random'
    CLUSTER = 'cluster'
    ONE_CLUSTER = 'one_cluster'


class Sampler:

    def __init__(self, k: int, models: Dict):
        self.k = k
        self.models = models

    def sample(self):
        raise NotImplementedError("not implemented")


class TopKSampler(Sampler):

    def __init__(self, model_scoring_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_scoring_fn = model_scoring_fn

    def sample(self):
        model_ids = list(self.models.keys())

        # we want to have model with highest score/performance first
        ranked_models = sorted(model_ids,
                               key=lambda model_id: self.model_scoring_fn(model_id),
                               reverse=True)
        return ranked_models[:self.k]


class RandomSampler(Sampler):
    def sample(self):
        selected_models = random.choices(list(self.models.keys()), k=self.k)
        return selected_models


def build_sampler(sampling_strategy: SamplingStrategy, num_models: int, models: Dict):
    default_args = dict(k=num_models, models=models)
    if sampling_strategy == SamplingStrategy.RANDOM:
        sampler = RandomSampler(**default_args)
    elif sampling_strategy == SamplingStrategy.CLUSTER:
        pass
    else:
        raise ValueError(f"Unknown sampling strategy. Possible values are {list(SamplingStrategy)}")
    return sampler


def main(num_models: int,
         sampling_strategies: List[SamplingStrategy],
         output_root: str,
         model_config_path: str,
         num_samples: int):
    # Retrieve model definitions
    with open(model_config_path, 'r') as f:
        models = json.load(f)

    os.makedirs(output_root, exist_ok=True)

    for sampling_strategy in sampling_strategies:
        sampler = build_sampler(sampling_strategy=sampling_strategy,
                                num_models=num_models,
                                models=models)

        model_sets = []
        for _ in range(num_samples):
            model_set = sampler.sample()
            model_sets.append(model_set)

        output_file = os.path.join(output_root, f'{sampling_strategy}.json')
        with open(output_file, 'w') as f:
            json.dump(model_sets, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_models', type=int)
    parser.add_argument('--sampling_strategies', nargs='+', choices=list(SamplingStrategy))
    parser.add_argument('--model_config_path', default='scripts/models_config.json')
    parser.add_argument('--num_samples')
    parser.add_argument('--output_root')
    args = parser.parse_args()

    main(**vars(args))
