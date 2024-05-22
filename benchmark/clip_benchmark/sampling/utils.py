import sqlite3
import pandas as pd
from functools import partial

RESULTS_PATH = ''


def retrieve_performance(model_id: str, dataset_id: str):
    conn = sqlite3.connect(RESULTS_PATH)
    df = pd.read_sql('SELECT * FROM "results"', conn)
    conn.close()
    # TODO


retrieve_imagenet_performance = partial(retrieve_performance())
