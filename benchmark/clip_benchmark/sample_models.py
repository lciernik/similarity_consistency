import os.path
from typing import Dict, List
import json
from enum import Enum
from functools import partial
import pandas as pd

from analysis.utils import retrieve_performance
from analysis.samplers import TopKSampler, RandomSampler, ClusterSampler, OneClusterSampler


class SamplingStrategy(Enum):
    TOP_K = 'top-k'
    RANDOM = 'random'
    CLUSTER_BEST = 'cluster_best'
    CLUSTER_RANDOM = 'cluster_random'
    ONE_CLUSTER = 'one_cluster'


def build_sampler(sampling_strategy: SamplingStrategy, num_models: int,
                  models: Dict, selection_dataset: str, cluster_assignment: Dict[int, List[str]]):
    # Returns the performance for a given model
    model_scoring_fn = partial(retrieve_performance, dataset_id=selection_dataset)

    default_args = dict(k=num_models, models=models)
    if sampling_strategy == SamplingStrategy.RANDOM:
        sampler = RandomSampler(**default_args)
    elif sampling_strategy == SamplingStrategy.TOP_K:
        sampler = TopKSampler(model_scoring_fn=model_scoring_fn, **default_args)
    elif sampling_strategy == SamplingStrategy.CLUSTER_BEST:
        sampler = ClusterSampler(cluster_assignment=cluster_assignment,
                                 selection_strategy='best',
                                 model_scoring_fn=model_scoring_fn,
                                 **default_args)
    elif sampling_strategy == SamplingStrategy.CLUSTER_RANDOM:
        sampler = ClusterSampler(cluster_assignment=cluster_assignment,
                                 selection_strategy='random',
                                 **default_args)
    elif sampling_strategy == SamplingStrategy.ONE_CLUSTER:
        sampler = OneClusterSampler(cluster_assignment=cluster_assignment,
                                    **default_args)
    else:
        raise ValueError(f"Unknown sampling strategy. Possible values are {list(SamplingStrategy)}")
    return sampler


def main(num_models: int,
         sampling_strategies: List[SamplingStrategy],
         output_root: str,
         model_config_path: str,
         selection_dataset: str,
         num_samples: int,
         cluster_assignment_path: str):
    # Retrieve model definitions
    with open(model_config_path, 'r') as f:
        models = json.load(f)

    cluster_assignment = None
    if cluster_assignment_path is not None:
        cluster_assignment = {}
        cluster_assignment_table = pd.read_csv(cluster_assignment_path)
        for cluster_idx in cluster_assignment_table.cluster.unique():
            subset = cluster_assignment_table[cluster_assignment_table.cluster == cluster_idx]
            # model names are in first column, we should give this column a name
            cluster_assignment[cluster_idx] = subset.iloc[:, 0].values.tolist()

    os.makedirs(output_root, exist_ok=True)

    for sampling_strategy in sampling_strategies:
        sampling_strategy = SamplingStrategy(sampling_strategy)
        sampler = build_sampler(sampling_strategy=sampling_strategy,
                                num_models=num_models,
                                models=models,
                                selection_dataset=selection_dataset,
                                cluster_assignment=cluster_assignment)

        model_sets = []
        n_sets = min(num_samples, sampler.max_available_samples())
        for _ in range(n_sets):
            model_set = sampler.sample()
            model_sets.append(model_set)

        if sampling_strategy in [SamplingStrategy.CLUSTER_RANDOM,
                                 SamplingStrategy.CLUSTER_BEST,
                                 SamplingStrategy.ONE_CLUSTER]:
            # cluster_assignment_path has the following structure
            # /home/space/diverse_priors/clustering/imagenet-subset-10k/{method}/num_clusters_{num_models}/cluster_labels.csv
            cluster_slug = cluster_assignment_path.split('/')[-3]
            output_file = os.path.join(output_root, f'{sampling_strategy.value}_{cluster_slug}.json')
        else:
            output_file = os.path.join(output_root, f'{sampling_strategy.value}.json')

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
    parser.add_argument('--cluster_assignment_path', type=str)
    parser.add_argument('--output_root', type=str)
    args = parser.parse_args()

    main(**vars(args))
