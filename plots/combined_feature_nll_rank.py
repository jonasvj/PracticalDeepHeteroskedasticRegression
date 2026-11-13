import warnings

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pypalettes import load_palette
from tueplots import bundles
from typing import List

from src.utils import parse_dirs
from src.utils.plot_utils import (
    COLOR_MAP_NAME,
    COLOR_MAP_TABLEAU_20,
    COLOR_MAP_TABLEAU_20_UMA,
    get_feature,
    map_feature_UMA,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def generate_df_qm9():
    """Load QM9 post-hoc data and compute per-feature NLL rank statistics."""
    directories = ['models/qm9']
    exclusions = ['models/qm9/wd_sweep', 'models/qm9/learning_curve']

    results = parse_dirs(directories, exclusions=exclusions)
    df = pd.DataFrame(results)

    df = df[df['stats_type'] == 'test']
    df = df.drop('stats_type', axis=1)
    df = df[df['method.name'] == 'post_hoc']

    df['feature'] = df['model.feature_extractor.module_indices_to_extract'].map(get_feature)
    df = df.drop('model.feature_extractor.module_indices_to_extract', axis=1)

    df.loc[
        df['model.variance_head._target_'] == 'src.models.painn.PainNNEnsembleVariancePrediction',
        'feature',
    ] = df['feature'].max(axis=None) + 1
    df = df.drop('model.variance_head._target_', axis=1)

    df['feature'] = pd.Categorical(df['feature'], np.sort(df['feature'].unique()))
    features = df['feature'].unique()

    if 'lpd' in df.columns and 'gaussian_nll' in df.columns:
        df.loc[df.feature == max(features), 'gaussian_nll'] = -df.loc[df.feature == max(features), 'lpd']

    assert df.shape[0] == 210, f"Expected 210 QM9 rows, got {df.shape[0]}"

    # Compute rank per (seed, target), then aggregate over all (seed, target) pairs
    metric = 'gaussian_nll'
    ranked_rows = []
    for seed in df['seed'].unique():
        for target in df['data.target'].unique():
            subset = df[(df['seed'] == seed) & (df['data.target'] == target)].copy()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                subset['rank'] = subset[metric].rank()
            ranked_rows.append(subset)

    combined_df = pd.concat(ranked_rows)
    avg_rank = combined_df.groupby('feature', observed=True)['rank'].mean().reset_index()
    sem_rank = combined_df.groupby('feature', observed=True)['rank'].sem().reset_index()
    return avg_rank, sem_rank


def generate_df_utkface():
    """Load UTKFace spatial post-hoc data and compute per-feature NLL statistics."""
    results = parse_dirs(
        directories=['models/utkface/post_hoc'],
        identifier=None,
        exclusions=None,
    )
    df = pd.DataFrame(results)

    df = df[df['stats_type'] == 'test']
    df = df.drop('stats_type', axis=1)

    df['feature'] = df['model.feature_extractor.module_indices_to_extract'].map(get_feature)
    df = df.drop('model.feature_extractor.module_indices_to_extract', axis=1)

    df.loc[
        df['model.variance_head._target_'] == 'src.models.resnet.ResNetEnsembleVariancePrediction',
        'feature',
    ] = df['feature'].max(axis=None) + 1
    df = df.drop('model.variance_head._target_', axis=1)

    df['feature'] = pd.Categorical(df['feature'], np.sort(df['feature'].unique()))
    features = df['feature'].unique()

    if 'lpd' in df.columns and 'gaussian_nll' in df.columns:
        df.loc[df.feature == max(features), 'gaussian_nll'] = -df.loc[df.feature == max(features), 'lpd']

    metric = 'gaussian_nll'
    mean_df = df.groupby('feature', observed=True)[metric].mean().reset_index()
    sem_df = df.groupby('feature', observed=True)[metric].sem().reset_index()
    return mean_df, sem_df


def generate_df_uma():
    """Load UMA post-hoc data and compute per-feature NLL statistics."""
    directories = ['models/uma']
    exclusions = ['models/uma/learning_curve']

    results = parse_dirs(directories, exclusions=exclusions)
    df = pd.DataFrame(results)

    df = df[df['stats_type'] == 'test']
    df = df.drop('stats_type', axis=1)
    df = df[df['method.name'] == 'post_hoc']

    df['feature'] = df['model.variance_head.feature_keys'].map(map_feature_UMA)
    df = df.drop('model.variance_head.feature_keys', axis=1)

    df.loc[
        df['model.variance_head._target_'] == 'src.models.FeatureModelEnsembleVariancePrediction',
        'feature',
    ] = df['feature'].max(axis=None) + 1
    df = df.drop('model.variance_head._target_', axis=1)

    # Remove homo rows (feature == -1)
    df = df[df['feature'] != -1]

    df['feature'] = pd.Categorical(df['feature'], np.sort(df['feature'].unique()))
    features = df['feature'].unique()

    if 'lpd' in df.columns and 'gaussian_nll' in df.columns:
        df.loc[df.feature == max(features), 'gaussian_nll'] = -df.loc[df.feature == max(features), 'lpd']

    metric = 'gaussian_nll'
    mean_df = df.groupby('feature', observed=True)[metric].mean().reset_index()
    sem_df = df.groupby('feature', observed=True)[metric].sem().reset_index()
    return mean_df, sem_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _feature_tick_labels(n_features: int) -> List[str]:
    """Return x-tick labels: z^0 ... z^{n-3}, All, Ensemble."""
    return [f'$z^{i}$' for i in range(n_features - 2)] + ['All', 'Ens.']


def _apply_panel_style(ax):
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.tick_params(axis='y', labelsize='small')
    ax.margins(y=0.15)
    sns.despine(ax=ax, top=True, right=True)


def make_plot(qm9_avg_rank, qm9_sem_rank, utkface_mean, utkface_sem, uma_mean, uma_sem):
    plt.rcParams.update(bundles.neurips2024(rel_width=1/3, nrows=1, ncols=1, usetex=True))
    plt.rcParams['axes.prop_cycle'] = mlp.cycler(color=load_palette(COLOR_MAP_NAME))

    # --- Figure 0: QM9 average NLL rank per feature ---
    fig, ax = plt.subplots(1, 1)
    features_q = qm9_avg_rank['feature'].to_numpy()
    y_q = qm9_avg_rank['rank'].to_numpy()
    y_err_q = qm9_sem_rank['rank'].to_numpy()

    for x, y, ye, f in zip(features_q, y_q, y_err_q, features_q):
        color = f'C{COLOR_MAP_TABLEAU_20[str(f)]}'
        ax.errorbar(x=x, y=y, yerr=ye, fmt='d', color=color, markersize=5)
    ax.set_ylabel(r'Avg. Rank $\downarrow$', fontsize='small')
    ax.set_xticks(features_q)
    labels = ax.set_xticklabels(_feature_tick_labels(len(features_q)), rotation=0, ha='center', fontsize='small')
    for label in labels[-2:]:
        label.set_rotation(45)
        label.set_ha('center')
    _apply_panel_style(ax)
    fig.savefig('plots/rank_nll_qm9.pdf', bbox_inches='tight')
    plt.close(fig)

    # --- Figure 1: UTKFace average NLL per feature ---
    fig, ax = plt.subplots(1, 1)
    features_u = utkface_mean['feature'].to_numpy()
    y_u = utkface_mean['gaussian_nll'].to_numpy()
    y_err_u = utkface_sem['gaussian_nll'].to_numpy()
    x_u = np.arange(len(features_u))

    for x, y, ye, f in zip(x_u, y_u, y_err_u, features_u):
        color = f'C{COLOR_MAP_TABLEAU_20[str(f)]}'
        ax.errorbar(x=x, y=y, yerr=ye, fmt='d', color=color, markersize=5)
    ax.set_ylabel(r'Avg. NLL $\downarrow$', fontsize='small')
    ax.set_xticks(x_u)
    labels = ax.set_xticklabels(_feature_tick_labels(len(features_u)), rotation=0, ha='center', fontsize='small')
    for label in labels[-2:]:
        label.set_rotation(45)
        label.set_ha('center')
    _apply_panel_style(ax)
    fig.savefig('plots/rank_nll_utkface.pdf', bbox_inches='tight')
    plt.close(fig)

    # --- Figure 2: UMA average NLL per feature ---
    fig, ax = plt.subplots(1, 1)
    features_m = uma_mean['feature'].to_numpy()
    y_m = uma_mean['gaussian_nll'].to_numpy()
    y_err_m = uma_sem['gaussian_nll'].to_numpy()
    x_m = np.arange(len(features_m))

    for x, y, ye, f in zip(x_m, y_m, y_err_m, features_m):
        color = f'C{COLOR_MAP_TABLEAU_20_UMA[str(f)]}'
        ax.errorbar(x=x, y=y, yerr=ye, fmt='d', color=color, markersize=5)
    ax.set_ylabel(r'Avg. NLL $\downarrow$', fontsize='small')
    ax.set_xticks(x_m)
    labels = ax.set_xticklabels(_feature_tick_labels(len(features_m)), rotation=0, ha='center', fontsize='small')
    for label in labels[-2:]:
        label.set_rotation(45)
        label.set_ha('center')
    _apply_panel_style(ax)
    fig.savefig('plots/rank_nll_uma.pdf', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    qm9_avg_rank, qm9_sem_rank = generate_df_qm9()
    utkface_mean, utkface_sem = generate_df_utkface()
    uma_mean, uma_sem = generate_df_uma()

    make_plot(qm9_avg_rank, qm9_sem_rank, utkface_mean, utkface_sem, uma_mean, uma_sem)
