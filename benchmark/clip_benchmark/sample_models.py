import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_models', type=int)
parser.add_argument('--sampling_strategy', choices=['top-k', 'random', 'cluster',
                                                    'one_cluster'])
parser.add_argument('--output_root')
args = parser.parse_args()


class Sampler:

    def __init__(self, k: int):
        self.k = k

    def sample(self):
        raise NotImplementedError("not implemented")
