import os
from pathlib import Path
from clip_benchmark.utils.utils import retrieve_model_dataset_results
import pandas as pd

HYPER_PARAM_COLS = ['model_ids', 'fewshot_k', 'fewshot_lr', 'fewshot_epochs', 'batch_size']


def retrieve_performance(model_id: str, dataset_id: str, metric_column: str = 'test_lp_acc1',
                         results_root='/home/space/diverse_priors/results/linear_probe/single_model'):
    path = os.path.join(results_root, dataset_id, model_id)

    df = retrieve_model_dataset_results(path)

    if df.dataset.nunique() > 1:
        raise ValueError(
            f"Result files for {model_id=} and {dataset_id=} contain multiple datasets. Cannot proceed."
        )
    performance = df.groupby(HYPER_PARAM_COLS)[metric_column].mean().max()
    return performance
