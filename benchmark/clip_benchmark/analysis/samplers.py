from typing import Dict, Callable
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
        selected_models = random.choices(list(self.models.keys()), k=self.k)
        return selected_models
