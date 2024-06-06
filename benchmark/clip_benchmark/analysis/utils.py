import os
import sqlite3

import pandas as pd

HYPER_PARAM_COLS = ['model_ids', 'fewshot_k', 'fewshot_lr', 'fewshot_epochs', 'batch_size']


def retrieve_performance(model_id: str, dataset_id: str, metric_column: str = 'test_lp_acc1',
                         results_root='/home/space/diverse_priors/results/linear_probe/single_model'):
    path = os.path.join(results_root, dataset_id, model_id, "results.db")
    conn = sqlite3.connect(path)
    df = pd.read_sql('SELECT * FROM "results"', conn)
    conn.close()
    subset = df[df.dataset == f'"{dataset_id}"']
    performance = subset.groupby(HYPER_PARAM_COLS)[metric_column].mean().max()
    return performance
