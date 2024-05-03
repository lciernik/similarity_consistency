from model_similarity import compute_sim_matrix
from feature_combiner import ConcatFeatureCombiner, PCAConcatFeatureCombiner


def get_feature_combiner_cls(combiner_name):
    if combiner_name == "concat":
        return ConcatFeatureCombiner
    elif combiner_name == "concat_pca":
        return PCAConcatFeatureCombiner
    else:
        raise ValueError(f"Unknown feature combiner name: {combiner_name}")
