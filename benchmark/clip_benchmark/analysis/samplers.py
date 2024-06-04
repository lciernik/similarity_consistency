from typing import Dict, Callable, List
import numpy as np
import random


class Sampler:

    def __init__(self, k: int, models: Dict):
        self.k = k
        self.models = models

    def sample(self):
        raise NotImplementedError("not implemented")

    def max_available_samples(self):
        return np.inf


class TopKSampler(Sampler):

    def __init__(self, model_scoring_fn: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_scoring_fn = model_scoring_fn

    def sample(self):
        model_ids = list(self.models.keys())

        # we want to have models with highest score/performance first
        ranked_models = sorted(model_ids,
                               key=lambda model_id: self.model_scoring_fn(model_id),
                               reverse=True)
        # take first k
        return ranked_models[:self.k]

    def max_available_samples(self):
        # this sampler always returns the same model set
        return 1


class RandomSampler(Sampler):
    """
    Samples totally random from the available models
    """

    def sample(self):
        selected_models = random.sample(list(self.models.keys()), k=self.k)
        return selected_models


class ClusterSampler(Sampler):

    def __init__(self, cluster_assignment: Dict[int, List[str]],
                 selection_strategy: str = 'random', model_scoring_fn: Callable = None
                 , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_assignment = cluster_assignment
        self.selection_strategy = selection_strategy
        if self.selection_strategy not in ['random', 'best']:
            raise ValueError('selection strategy should be either random or best')

        if self.selection_strategy == 'best' and model_scoring_fn is None:
            raise ValueError('model_scoring_fn needs to be specified')

        self.model_scoring_fn = model_scoring_fn

    def sample(self):
        available_clusters = list(self.cluster_assignment.keys())
        if len(available_clusters) < self.k:
            raise ValueError('num_cluster should be larger than k')

        selected_clusters = random.sample(available_clusters, k=self.k)

        model_set = []
        for cluster in selected_clusters:
            if self.selection_strategy == 'random':
                selected_model = random.choice(self.cluster_assignment[cluster])
            elif self.selection_strategy == 'best':
                ranked_models = sorted(self.cluster_assignment[cluster],
                                       key=lambda model_id: self.model_scoring_fn(model_id),
                                       reverse=True)
                selected_model = ranked_models[0]
            else:
                raise ValueError("unknown model selection strategy")
            model_set.append(selected_model)
        return model_set


class OneClusterSampler(Sampler):

    def __init__(self, cluster_assignment: Dict[int, List[str]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_assignment = cluster_assignment

    def sample(self):
        available_clusters = list(self.cluster_assignment.keys())
        selected_cluster = random.choice(available_clusters)
        model_options = self.cluster_assignment[selected_cluster]
        model_set = random.sample(model_options, k=min(self.k, len(model_options)))
        return model_set
