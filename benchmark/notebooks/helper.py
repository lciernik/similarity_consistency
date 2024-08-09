from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_model_ids(fn):
    with open(fn, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    return lines


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
    );
    sns.histplot(
        r_values,
        x='r_coeff',
        hue=sim_met_col,
        bins=10,
        multiple='dodge',
        kde=True,
        ax=axs[1],
        alpha=0.5,

    );
    sns.kdeplot(
        r_values,
        x='r_coeff',
        hue=sim_met_col,
        ax=axs[2]
    );

    for i in range(3):
        axs[i].set_xlabel('Correlation coefficient');

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
