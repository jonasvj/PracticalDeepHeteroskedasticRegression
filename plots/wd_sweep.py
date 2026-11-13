import pandas as pd
import numpy as np
from typing import List
import matplotlib as mlp
from tueplots import bundles
from src.utils import parse_dirs
import matplotlib.pyplot as plt
from pypalettes import load_palette
from src.utils.plot_utils import (
    get_feature, COL_TO_LABEL, FEATURE_IDX_TO_LABEL, COLOR_MAP_TABLEAU_20
)


def main(
    data: dict,
    columns: List[str],
    metrics: List[str],
    target: int,
    plot_name: str
):
    results = []
    for value in data.values():
        results.extend(
            parse_dirs(
                directories=value['dir'],
                identifier=value['identifier'],
                exclusions=value['exclusions'],
            )
        )
    df = pd.DataFrame(results)
    df = df[columns + list(set([*metrics]))]
    x_axis = 'method.calibrate.optimizer.weight_decay'

    # Keep only selected target
    df = df[df['data.target'] == target]

    # Keep only test results
    df = df[df['stats_type'] == 'test']
    df = df.drop('stats_type', axis=1)

    # Simplify feature variable
    df['feature'] = df['model.feature_extractor.module_indices_to_extract'].map(get_feature)
    df = df.drop('model.feature_extractor.module_indices_to_extract', axis=1)
    
    # Set feature for ensemble
    df.loc[
        df['model.variance_head._target_'] == 'src.models.painn.PainNNEnsembleVariancePrediction', 
        'feature'
    ] = df['feature'].max(axis=None) + 1
    df = df.drop('model.variance_head._target_', axis=1)

    df['feature'] = pd.Categorical(
        df['feature'], np.sort(df['feature'].unique())
    )
    features = df['feature'].unique()
    FEATURE_IDX_TO_LABEL[max(features) - 1] = r'$\textrm{All}$'
    FEATURE_IDX_TO_LABEL[max(features)] = r'$\textrm{Ensemble}$'

    # Check that all results are there
    counts = df.groupby(['data.target', 'feature', x_axis], dropna=False)[metrics[0]].count()
    with pd.option_context("display.max_rows", None):
        print(counts)
    assert sum(counts == 3) == len(counts)

    if 'lpd' in df.columns and 'gaussian_nll' in df.columns:
        df.loc[df.feature == max(features), 'gaussian_nll'] = -df.loc[df.feature == max(features), 'lpd']
        df = df.drop('lpd', axis=1)
        metrics = [m for m in metrics if m != 'lpd']
    
    if 'ece_mixture' in df.columns and 'ece' in df.columns:
        df.loc[df.feature == max(features), 'ece'] = df.loc[df.feature == max(features), 'ece_mixture']
        df = df.drop('ece_mixture', axis=1)
        metrics = [m for m in metrics if m != 'ece_mixture']
    
    with pd.option_context("display.max_rows", None):
        print(df)

    # Average over seeds
    df = df.drop('seed', axis=1)
    groupby_columns = list(set(df.columns) - set(['seed', *metrics]))
    mean = df.groupby(groupby_columns, observed=True).mean().reset_index()
    sem = df.groupby(groupby_columns, observed=True).sem().reset_index()

    plt.rcParams.update(bundles.neurips2024(rel_width=0.5, nrows=1, ncols=1, usetex=True))
    plt.rcParams['axes.prop_cycle'] = mlp.cycler(color=load_palette("Tableau_20")) 

    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)

    features = mean['feature'].unique()
    n_features = len(features)
 
    unique_x = np.sort(mean[x_axis].unique())
    # log-bin edges halfway between neighboring x values
    segments = np.empty(len(unique_x) + 1)
    segments[1:-1] = np.sqrt(unique_x[:-1] * unique_x[1:])

    # extrapolate outer edges
    segments[0]  = unique_x[0] / np.sqrt(unique_x[1] / unique_x[0])
    segments[-1] = unique_x[-1] * np.sqrt(unique_x[-1] / unique_x[-2])

    bin_widths = segments[1:] / segments[:-1]
    
    for feature in features:
        mean_tmp = mean[mean['feature'] == feature].sort_values(by=x_axis)
        sem_tmp = sem[sem['feature'] == feature].sort_values(by=x_axis)

        offsets = np.linspace(-0.35, 0.35, n_features)
        feature_to_offset = dict(zip(features, offsets))

        x = mean_tmp[x_axis].values
        idx = np.searchsorted(unique_x, x)
        x_jittered = x * (bin_widths[idx] ** feature_to_offset[feature])

        # Version 2
        ax[0, 0].errorbar(
            x_jittered,
            mean_tmp[metrics[0]],
            yerr=sem_tmp[metrics[0]],
            label=FEATURE_IDX_TO_LABEL[feature],
            linestyle=None,
            elinewidth=1,
            fmt='d',
            color=f'C{COLOR_MAP_TABLEAU_20[str(feature)]}',
            markersize=3,
        )

    ax[0,0].spines['top'].set_visible(False)
    ax[0,0].spines['right'].set_visible(False)
    ax[0,0].spines['bottom'].set_visible(False)
    ax[0,0].spines['left'].set_visible(False)
    ax[0,0].grid(True, axis='y', zorder=1, color='grey', linewidth=0.5)
    ax[0,0].set_axisbelow(True)
    ax[0,0].set_xlabel(COL_TO_LABEL[x_axis])
    ax[0,0].set_ylabel(COL_TO_LABEL[metrics[0]])
    ax[0,0].set_xscale('log')
    ax[0,0].tick_params(axis='x', which='both', bottom=False, top=False)
    ax[0,0].tick_params(axis='y', color='grey', width=0.5)

    ax[0,0].xaxis.set_label_position("top")  # label above ticks

    # Move label left or right
    ax[0,0].xaxis.set_label_coords(-0.05, -0.15)   # right of ticks

    ylims = ax[0,0].get_ylim()

    for i in range(len(segments) - 1):
        alpha = 0.2 if i % 2 == 0 else 0.1
        col ='grey'
        ax[0,0].fill_betweenx(ylims, segments[i], segments[i+1], color=col, alpha=alpha, zorder=-3, edgecolor=None)

    ax[0,0].set_xlim([segments[0], segments[-1]])
    ax[0,0].set_ylim(ylims)

    handles, labels = ax[0,0].get_legend_handles_labels()

    # first row (5 items)
    leg1 = ax[0,0].legend(
        handles[:5], labels[:5],
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        frameon=False
    )
    ax[0,0].add_artist(leg1)
    # second row (2 items)
    ax[0,0].legend(
        handles[5:], labels[5:],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        frameon=False
    )

    fig.savefig(plot_name)


if __name__ == '__main__':
    target = 8
    data = {
        0: {
            'dir': ['models/qm9/wd_sweep'],
            'identifier': 'post_hoc',
            'exclusions': ['/wd-1e-2/', '_mlp']
        },
        1: {
            'dir': [f'models/qm9/target-{target}/post_hoc/'],
            'identifier': 'post_hoc',
            'exclusions': None,
        },
    }
    columns = [
        'stats_type',
        'seed',
        'data.target',
        'model.feature_extractor.module_indices_to_extract',
        'method.calibrate.optimizer.weight_decay',
        'model.variance_head._target_'
    ]
    metrics = [
        'lpd',
        'gaussian_nll',
    ]
    plot_name = f'plots/wd_sweep_qm9_target_{target}.pdf'
    main(data, columns, metrics, target, plot_name)