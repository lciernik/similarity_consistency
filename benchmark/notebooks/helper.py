import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Union, List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from notebooks.constants import ds_info_file
from scripts.helper import parse_datasets


def plot_r_coeff_distribution(df, sim_met_col, r_x, r_y='gap', ds_col='dataset'):
    r_vals = []
    for key, group_data in df.groupby([ds_col, sim_met_col]):
        r = group_data[r_x].corr(group_data[r_y], method="spearman")
        r_vals.append({
            'Dataset': key[0],
            sim_met_col: key[1],
            'r_coeff': r,
        })

    r_values = pd.DataFrame(r_vals)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    sns.boxplot(
        r_values,
        x=sim_met_col,
        y='r_coeff',
        ax=axs[0],
    )
    sns.histplot(
        r_values,
        x='r_coeff',
        hue=sim_met_col,
        bins=10,
        multiple='dodge',
        kde=True,
        ax=axs[1],
        alpha=0.5,

    )
    sns.kdeplot(
        r_values,
        x='r_coeff',
        hue=sim_met_col,
        ax=axs[2]
    )

    for i in range(3):
        axs[i].set_xlabel('Correlation coefficient')

    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha="right")

    fig.suptitle(f'Distibution correlation coefficients over all datasets.')

    vmin = max(r_values['r_coeff'].min(), -0.5)
    vmax = min(r_values['r_coeff'].max(), 0.5)

    for idx, val in product([1, 2], [vmin, vmax]):
        axs[idx].axvline(val, ls=':', c='grey', alpha=0.5)

    return fig


def plot_scatter(df, title, ds, sim_met_col, sim_val_col):
    g = sns.relplot(
        df,
        x=sim_val_col,
        y='gap',
        col=sim_met_col,
        row='dataset',
        height=3,
        aspect=1.25,
        facet_kws={'sharey': False, 'sharex': False}
    )
    g.set_titles("{row_name} – {col_name}")
    g.set_ylabels("Performance gap combined - anchor")
    g.set_xlabels(f"Similarity value {ds}.")

    def annotate_correlation(data, **kwargs):
        r = data[sim_val_col].corr(data['gap'], method="spearman")
        ax = plt.gca()
        ax.text(0.05, 0.95, f'r = {r:.2f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top')
        if max(data['gap']) > 0:
            ax.axhspan(0, max(data['gap']), facecolor='lightgreen', alpha=0.2, zorder=-1)
        if min(data['gap']) < 0:
            ax.axhspan(min(data['gap']), 0, facecolor='lightcoral', alpha=0.2, zorder=-1)

    g.map_dataframe(annotate_correlation)

    g.fig.suptitle(title, y=1)
    g.fig.tight_layout()
    return g.fig


def save_or_show(fig, path, save):
    if save:
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f'stored img at {path}.')
    else:
        plt.show(fig)


def get_model_ids(fn: Union[str, Path]) -> List[str]:
    """
    Load model ids from file.
    Args:
        fn: Path to file containing model ids.

    Returns:
        List of model ids.
    """
    with open(fn, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    return lines


def load_sim_matrix(path: Union[str, Path], allowed_models: List[str]) -> pd.DataFrame:
    """
    Load similarity matrix from file and filter for allowed models.
    Args:
        path: Path to similarity matrix.
        allowed_models: List of allowed model ids.

    Returns:

    """
    model_ids_fn = path / 'model_ids.txt'
    sim_mat_fn = path / 'similarity_matrix.pt'
    if model_ids_fn.exists():
        model_ids = get_model_ids(model_ids_fn)
    else:
        raise FileNotFoundError(f'{str(model_ids_fn)} does not exist.')
    sim_mat = torch.load(sim_mat_fn)
    df = pd.DataFrame(sim_mat, index=model_ids, columns=model_ids)
    df = df.loc[allowed_models, allowed_models]
    return df


def load_similarity_matrices(
        path: Union[str, Path],
        ds_list: List[str],
        sim_metrics: List[str],
        allowed_models: List[str]
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load similarity matrices for all datasets and similarity metrics.
    Args:
        ds_list: List of dataset names
        sim_metrics: List of similarity metrics
        path: Base path to similarity matrices
        allowed_models: List of allowed model ids

    Returns:
        Dictionary of similarity matrices for all similarity metrics and datasets. Structure: {sim_metric: {ds: pd.DataFrame}}

    """
    sim_mats = defaultdict(dict)
    for sim_metric in sim_metrics:
        for ds in ds_list:
            sim_mats[sim_metric][ds] = load_sim_matrix(path / ds / sim_metric, allowed_models)
            np.fill_diagonal(sim_mats[sim_metric][ds].values, 1)
    return sim_mats


def load_model_configs_and_allowed_models(
        path: Union[str, Path],
        exclude_models: List[str] = ['SegmentAnything_vit_b', 'DreamSim_dino_vitb16', 'DreamSim_open_clip_vitb32'],
        exclude_alignment: bool = True,
        sort_by: str = 'objective',

) -> Tuple[pd.DataFrame, List[str]]:
    with open(path, 'r') as f:
        model_configs = json.load(f)

    print(f"Nr. models original={len(model_configs)}")
    models_to_exclude = [k for k, v in model_configs.items() if
                         (exclude_alignment and v['alignment'] is not None) or (k in exclude_models)]
    if models_to_exclude:
        for k in models_to_exclude:
            model_configs.pop(k)
        print(f"Nr. models after exclusion={len(model_configs)}")

    model_configs = pd.DataFrame(model_configs).T
    model_configs = model_configs.reset_index().sort_values([sort_by, 'index']).set_index('index')

    allowed_models = model_configs.index.tolist()

    return model_configs, allowed_models


def load_ds_info(path):
    with open(path, 'r') as f:
        ds_info = json.load(f)
    ds_info = {k.replace('/', '_'): v for k, v in ds_info.items()}
    ds_info = pd.DataFrame(ds_info).T
    return ds_info


def load_all_datasetnames_n_info(path, verbose=False):
    ds_list = parse_datasets(path)
    ds_list = list(map(lambda x: x.replace('/', '_'), ds_list))
    if verbose:
        print(ds_list, len(ds_list))

    ds_info = load_ds_info(ds_info_file)
    return ds_list, ds_info


def get_fmt_name(ds, ds_info):
    return ds_info.loc[ds]['name'] + ' (' + ds_info.loc[ds]['domain'] + ')'


def pp_storing_path(path, save):
    if not isinstance(path, Path):
        path = Path(path)
    if save:
        path.mkdir(parents=True, exist_ok=True)
        print()
    return path
