import os
import sqlite3

import pandas as pd

HYPER_PARAM_COLS = ['model_ids', 'fewshot_k', 'fewshot_lr', 'fewshot_epochs', 'batch_size']


def retrieve_performance(model_id: str, dataset_id: str, metric_column: str = 'test_lp_acc1',
                         results_root='/home/space/diverse_priors/results/linear_probe/single_model'):
    path = os.path.join(results_root, dataset_id, model_id, "results.db")
    try:
        conn = sqlite3.connect(path)
        df = pd.read_sql('SELECT * FROM "results"', conn)
        conn.close()
    except pd.errors.DatabaseError as e:
        # TODO this is temporary
        print(f"Tried to extract data from {path=}, but got Error: {e}")
        print(f"{os.path.exists(path)=}")
        raise e
    if df.dataset.nunique() > 1:
        raise ValueError(
            f"Database at {path} contains results of multiple datasets. "
            f"Maybe something went wrong before ..."
        )
    performance = df.groupby(HYPER_PARAM_COLS)[metric_column].mean().max()
    return performance