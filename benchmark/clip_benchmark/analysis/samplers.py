import random
import warnings
from typing import Dict, Callable, List

import numpy as np


class Sampler:

    def __init__(self, k: int, models: Dict, seed: int = 0):
        self.k = k
        self.models = models
        random.seed(seed)

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

        # we want to have models with the highest score/performance first
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


class BaseClusterSampler(Sampler):

    def __init__(self,
                 cluster_assignment: Dict[int, List[str]],
                 model_scoring_fn: Callable = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_assignment = cluster_assignment
        self.model_scoring_fn = model_scoring_fn

    def _get_mean_cluster_score(self, cluster_id: int):
        if self.model_scoring_fn is None:
            raise ValueError('model_scoring_fn needs to be specified')
        cluster_models = self.cluster_assignment[cluster_id]
        cluster_scores = [self.model_scoring_fn(model_id) for model_id in cluster_models]
        return np.mean(cluster_scores)

    def rank_clusters_by_mean_score(self):
        cluster_scores = {cluster_id: self._get_mean_cluster_score(cluster_id) for cluster_id in
                          self.cluster_assignment.keys()}
        ranked_cluster_ids = sorted(cluster_scores, key=lambda cluster_id: cluster_scores[cluster_id], reverse=True)
        return ranked_cluster_ids

    def sample(self):
        raise NotImplementedError("Method `sample` not implemented for BaseClusterSampler. Use subclasses instead.")


class ClusterSampler(BaseClusterSampler):

    def __init__(self, selection_strategy: str = 'random', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection_strategy = selection_strategy
        if self.selection_strategy not in ['random', 'best']:
            raise ValueError('selection strategy should be either random or best')

        if self.selection_strategy == 'best' and self.model_scoring_fn is None:
            raise ValueError('model_scoring_fn needs to be specified')

    def get_selected_clusters(self) -> List[int]:
        if len(self.cluster_assignment) < self.k:
            raise ValueError('ClusterSampler requires at least k clusters in self.cluster_assignment to work.')
        ranked_cluster_ids = self.rank_clusters_by_mean_score()
        selected_clusters = ranked_cluster_ids[:self.k]
        return selected_clusters

    def sample(self) -> List[str]:
        selected_clusters = self.get_selected_clusters()
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

    def max_available_samples(self):
        return np.inf if self.selection_strategy == 'random' else 1


class OneClusterSampler(BaseClusterSampler):

    def get_selected_cluster(self) -> int:
        if len(self.cluster_assignment) == 0:
            raise ValueError('No cluster found.')

        ranked_cluster_ids = self.rank_clusters_by_mean_score()
        # Filter out clusters with less than k models
        selected_clusters = [cluster for cluster in ranked_cluster_ids if
                             len(self.cluster_assignment[cluster]) >= self.k]
        if len(selected_clusters) == 0:
            raise ValueError('no cluster with at least k models found')
        return selected_clusters[0]

    def sample(self) -> List[str]:
        selected_cluster = self.get_selected_cluster()
        model_options = self.cluster_assignment[selected_cluster]
        if self.k > len(model_options):
            warnings.warn('num of selected models is larger than cluster size, limiting to cluster size..')
        model_set = random.sample(model_options, k=min(self.k, len(model_options)))
        return model_set
