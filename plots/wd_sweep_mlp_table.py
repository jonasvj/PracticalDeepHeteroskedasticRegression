import numpy as np
import pandas as pd
from typing import List
import matplotlib as mlp
from tueplots import bundles
from src.utils import parse_dirs
import matplotlib.pyplot as plt
from pypalettes import load_palette
from src.utils.plot_utils import (
    get_feature, COL_TO_LABEL,
    FEATURE_IDX_TO_LABEL, COLOR_MAP_TABLEAU_20,
)


def main(
    data: dict,
    columns: List[str],
    metrics: List[str],
    targets: List[int],
    plot_name: str,
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

    # Keep only selected targets
    df = df[df['data.target'].isin(targets)]

    # Keep only test results
    df = df[df['stats_type'] == 'test']
    df = df.drop('stats_type', axis=1)

    # Tag MLP results before dropping the _target_ column
    is_mlp = df['model.variance_head._target_'] == 'src.models.painn.PaiNNMLPVariancePrediction'

    # Simplify feature variable
    df['feature'] = df['model.feature_extractor.module_indices_to_extract'].map(get_feature)
    df = df.drop('model.feature_extractor.module_indices_to_extract', axis=1)

    # Set feature for ensemble
    df.loc[
        df['model.variance_head._target_'] == 'src.models.painn.PainNNEnsembleVariancePrediction',
        'feature'
    ] = df['feature'].max(axis=None) + 1

    # Assign MLP results a separate feature index (before making Categorical)
    mlp_feature = int(df['feature'].max(axis=None)) + 1
    df.loc[is_mlp, 'feature'] = mlp_feature

    df = df.drop('model.variance_head._target_', axis=1)

    df['feature'] = pd.Categorical(
        df['feature'], np.sort(df['feature'].unique())
    )

    FEATURE_IDX_TO_LABEL[mlp_feature - 2] = r'$\textrm{All}$'
    FEATURE_IDX_TO_LABEL[mlp_feature - 1] = r'$\textrm{Ensemble}$'
    FEATURE_IDX_TO_LABEL[mlp_feature] = r'$\textrm{All (MLP)}$'
    COLOR_MAP_TABLEAU_20[str(mlp_feature)] = 8

    # Handle ensemble metrics
    if 'lpd' in df.columns and 'gaussian_nll' in df.columns:
        ensemble_feature = mlp_feature - 1
        df.loc[df.feature == ensemble_feature, 'gaussian_nll'] = -df.loc[df.feature == ensemble_feature, 'lpd']
        df = df.drop('lpd', axis=1)
        metrics = [m for m in metrics if m != 'lpd']

    if 'ece_mixture' in df.columns and 'ece' in df.columns:
        ensemble_feature = mlp_feature - 1
        df.loc[df.feature == ensemble_feature, 'ece'] = df.loc[df.feature == ensemble_feature, 'ece_mixture']
        df = df.drop('ece_mixture', axis=1)
        metrics = [m for m in metrics if m != 'ece_mixture']

    # Keep only All, Ensemble, and All (MLP)
    keep_features = [mlp_feature - 2, mlp_feature - 1, mlp_feature]
    df = df[df['feature'].isin(keep_features)]
    df['feature'] = pd.Categorical(df['feature'], keep_features)

    # Average over seeds
    df = df.drop('seed', axis=1)
    groupby_columns = list(set(df.columns) - set(['seed', *metrics]))
    mean = df.groupby(groupby_columns, observed=True).mean().reset_index()
    sem = df.groupby(groupby_columns, observed=True).sem().reset_index()

    features = mean['feature'].unique()
    n_features = len(features)

    unique_x = np.sort(mean[x_axis].unique())
    # log-bin edges halfway between neighbouring x values
    segments = np.empty(len(unique_x) + 1)
    segments[1:-1] = np.sqrt(unique_x[:-1] * unique_x[1:])
    segments[0] = unique_x[0] / np.sqrt(unique_x[1] / unique_x[0])
    segments[-1] = unique_x[-1] * np.sqrt(unique_x[-1] / unique_x[-2])
    bin_widths = segments[1:] / segments[:-1]

    offsets = np.linspace(-0.35, 0.35, n_features)
    feature_to_offset = dict(zip(features, offsets))

    for target in targets:
        plt.rcParams.update(bundles.neurips2024(rel_width=0.5, nrows=1, ncols=1, usetex=True))
        plt.rcParams['axes.prop_cycle'] = mlp.cycler(color=load_palette("Tableau_20"))

        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)

        mean_t = mean[mean['data.target'] == target]
        sem_t = sem[sem['data.target'] == target]

        for feature in features:
            mean_tmp = mean_t[mean_t['feature'] == feature].sort_values(by=x_axis)
            sem_tmp = sem_t[sem_t['feature'] == feature].sort_values(by=x_axis)

            x = mean_tmp[x_axis].values
            idx = np.searchsorted(unique_x, x)
            x_jittered = x * (bin_widths[idx] ** feature_to_offset[feature])

            ax.errorbar(
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

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(True, axis='y', zorder=1, color='grey', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xscale('log')
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ax.tick_params(axis='y', color='grey', width=0.5)

        ax.xaxis.set_label_position('top')
        ax.set_xlabel(COL_TO_LABEL[x_axis])
        ax.xaxis.set_label_coords(-0.05, -0.15)
        ax.set_ylabel(COL_TO_LABEL[metrics[0]])

        ylims = ax.get_ylim()

        for i in range(len(segments) - 1):
            alpha = 0.2 if i % 2 == 0 else 0.1
            ax.fill_betweenx(ylims, segments[i], segments[i + 1], color='grey', alpha=alpha, zorder=-3, edgecolor=None)

        ax.set_xlim([segments[0], segments[-1]])
        ax.set_ylim(ylims)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels,
            ncol=n_features,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.25),
            frameon=False,
        )

        target_plot_name = plot_name.replace('.pdf', f'_target_{target}.pdf')
        fig.savefig(target_plot_name, bbox_inches='tight')
        print(f'Plot written to {target_plot_name}')
        plt.close(fig)


if __name__ == '__main__':
    targets = [7, 8]
    data = {
        0: {
            'dir': ['models/qm9/wd_sweep'],
            'identifier': 'post_hoc',
            'exclusions': ['/wd-1e-2/', '_mlp']
        },
        1: {
            'dir': [f'models/qm9/target-{t}/post_hoc/' for t in targets],
            'identifier': 'post_hoc',
            'exclusions': None,
        },
        2: {
            'dir': ['models/qm9/wd_sweep'],
            'identifier': 'feature-all_mlp',
            'exclusions': []
        },
    }
    columns = [
        'stats_type',
        'seed',
        'data.target',
        'model.feature_extractor.module_indices_to_extract',
        'method.calibrate.optimizer.weight_decay',
        'model.variance_head._target_',
    ]
    metrics = [
        'lpd',
        'gaussian_nll',
    ]
    plot_name = 'plots/wd_sweep_mlp_table_qm9.pdf'
    main(data, columns, metrics, targets, plot_name)
