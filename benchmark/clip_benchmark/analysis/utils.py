import os
from pathlib import Path
from clip_benchmark.utils.utils import retrieve_model_dataset_results
import pandas as pd

HYPER_PARAM_COLS = ['model_ids', 'fewshot_k', 'fewshot_epochs', 'batch_size']


def retrieve_performance(model_id: str, dataset_id: str, metric_column: str = 'test_lp_acc1',
                         results_root: str ='/home/space/diverse_priors/results/linear_probe/single_model',
                         regularization:str = "weight_decay"):
    
    path = os.path.join(results_root, dataset_id, model_id)

    df = retrieve_model_dataset_results(path)

    if df.dataset.nunique() > 1:
        raise ValueError(
            f"Result files for {model_id=} and {dataset_id=} contain multiple datasets. Cannot proceed."
        )

    # filter regularization method
    df = df[df.regularization == regularization]
    if len(df) == 0: 
        raise ValueError(f'No results available for {dataset=}, {model_id=} and {regularization=}.')

    performance = df.groupby(HYPER_PARAM_COLS)[metric_column].mean().max()
    return performance
