import pandas as pd
from typing import List
from src.utils import parse_dirs
from tueplots import bundles
import matplotlib.pyplot as plt
import matplotlib as mlp
from matplotlib.patches import Patch
from pypalettes import load_palette
from src.utils.plot_utils import (
    COL_TO_LABEL, FEATURE_IDX_TO_LABEL, COLOR_MAP_TABLEAU_20,
    COLOR_MAP_TABLEAU_20_UMA, COLOR_MAP_NAME, METHOD_TO_LABEL, map_feature_UMA
)

def generate_df(
    data: dict,
    columns: List[str],
    metrics: List[str],
    val_metric: str,
    methods: List[str],
    target: int,
    ):  
    results = []
    for value in data.values():
        results.extend(
            parse_dirs(
                directories=[value['dir']],
                identifier=value['identifier'],
                exclusions=value['exclusions'],
            )
        )
    df = pd.DataFrame(results)

    # Ensemble checkpoints are missing data.cal_subset_size — infer from model_dir
    mask = df['data.cal_subset_size'].isna() & df['model_dir'].str.contains('subset-')
    df.loc[mask, 'data.cal_subset_size'] = df.loc[mask, 'model_dir'].str.extract(r'subset-([\d.]+)')[0].astype(float).values

    df = df[columns + list(set([*metrics, val_metric]))]
    df = df[df['method.name'].isin(methods)]
    
        
    # Keep only selected target
    df = df[df['data.target'] == target]
    # Keep only test results
    df = df[df['stats_type'] == 'test']
    df = df.drop('stats_type', axis=1)

    # Simplify feature variable.
    df['feature'] = df['model.variance_head.feature_keys'].map(map_feature_UMA)
    df = df.drop('model.variance_head.feature_keys', axis=1)

    # Set feature for ensemble.
    df.loc[
        df['model.variance_head._target_'] == 'src.models.FeatureModelEnsembleVariancePrediction', 
        'feature'
    ] = df['feature'].max(axis=None) + 1
    df = df.drop('model.variance_head._target_', axis=1)

    
    df['feature'] = pd.Categorical(
        df['feature'], df['feature'].unique().sort()
    )

    features = df['feature'].unique()

    # Convert NLL and ECE of ensemble to -lpd and ece_mixture
    if 'lpd' in df.columns and 'gaussian_nll' in df.columns:
        df.loc[df.feature == max(features), 'gaussian_nll'] = -df.loc[df.feature == max(features), 'lpd']
        metrics = [m for m in metrics if m != 'lpd']
    
    if 'ece_mixture' in df.columns and 'ece' in df.columns:
        df.loc[df.feature == max(features), 'ece'] = df.loc[df.feature == max(features), 'ece_mixture']
        metrics = [m for m in metrics if m != 'ece_mixture']
   
    return df


def gen_individual_plots(df, metric_to_plot="gaussian_nll", num_cal_points_full=10000):
    
    # Average over seeds
    df = df.drop('seed', axis=1)
    groupby_columns = list(set(df.columns) - set(['seed', *metrics]))
    mean = df.groupby(groupby_columns, observed=True).mean().reset_index()
    sem = df.groupby(groupby_columns, observed=True).sem().reset_index()
    x_axis = 'data.cal_subset_size'

    plt.rcParams.update(bundles.neurips2024(rel_width=0.5, nrows=1, ncols=1, usetex=True))
    plt.rcParams['axes.prop_cycle'] = mlp.cycler(color=load_palette(COLOR_MAP_NAME)) 

    plt.rcParams['figure.constrained_layout.use'] = False
    fig, ax= plt.subplots(nrows=1, ncols=1, squeeze=False)
    fig.subplots_adjust(bottom=0.425, top=0.99, left=0.15, right=0.95)

    features = mean['feature'].unique()
    methods = [m for m in mean['method.name'].unique()]

    FEATURE_IDX_TO_LABEL[max(features) - 1] = r'$\textrm{All}$'
    FEATURE_IDX_TO_LABEL[max(features)] = r'$\textrm{Ensemble}$'

    for i, method in enumerate(methods):
        if method == "post_hoc":
            mean_tmp = mean[(mean['method.name'] == method) & (mean['feature'] == max(features))].sort_values(by=x_axis)
            sem_tmp = sem[(sem['method.name'] == method) & (sem['feature'] == max(features))].sort_values(by=x_axis)            
        else:
            mean_tmp = mean[mean['method.name'] == method].sort_values(by=x_axis)
            sem_tmp = sem[sem['method.name'] == method].sort_values(by=x_axis)  
        ax[0,0].errorbar(
            mean_tmp[x_axis]*num_cal_points_full,
            mean_tmp[metric_to_plot],
            yerr=sem_tmp[metric_to_plot],
            label=METHOD_TO_LABEL[method],
            color=f'C{COLOR_MAP_TABLEAU_20[method]}',
        )

    ax[0,0].spines['top'].set_visible(False)
    ax[0,0].spines['right'].set_visible(False)
    ax[0,0].grid(True, zorder=-1, color='grey', linewidth=0.5)
    ax[0,0].set_axisbelow(True)
    ax[0,0].set_xscale('log', base=10)

    ax[0,0].set_ylabel(COL_TO_LABEL[metric_to_plot])
    ax[0,0].set_xlabel("No. Hold-out Data")
    handles, labels = ax[0,0].get_legend_handles_labels()
    handles = handles + [handles[0]]*4
    labels = labels + [labels[0]]*4

    leg1 = ax[0,0].legend(
        [Patch(alpha=0)] * len(handles[3:]), labels[3:],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        frameon=False,
        handlelength=0,
        markerscale=0,
    )
    ax[0,0].add_artist(leg1)
    # second row (2 items)
    ax[0,0].legend(
        handles[:2], labels[:2],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.45),
        frameon=False
    )
    ax[0,0].add_artist(leg1)
    
    for text in leg1.get_texts():
        text.set_color('white')
        text.set_alpha(0)

    for h in leg1.legend_handles:
        h.set_alpha(0)

    fig.savefig("plots/learning_curve_uma_methods.pdf")


    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)

    features = list(range(max(mean['feature'].unique()) + 1))
    for feature in features:
        mean_tmp = mean[(mean['feature'] == feature) & (mean['method.name'] == "post_hoc")].sort_values(by=x_axis)
        sem_tmp = sem[(sem['feature'] == feature) & (sem['method.name'] == "post_hoc")].sort_values(by=x_axis)
        ax[0,0].errorbar(
            mean_tmp[x_axis]*num_cal_points_full,
            mean_tmp[metric_to_plot],
            yerr=sem_tmp[metric_to_plot],
            label=FEATURE_IDX_TO_LABEL[feature],
            color=f'C{COLOR_MAP_TABLEAU_20_UMA[str(feature)]}',
        )

    ax[0,0].spines['top'].set_visible(False)
    ax[0,0].spines['right'].set_visible(False)
    ax[0,0].grid(True, zorder=-1, color='grey', linewidth=0.5)
    ax[0,0].set_axisbelow(True)
    ax[0,0].set_xscale('log', base=10)

    ax[0,0].set_ylabel(COL_TO_LABEL[metric_to_plot])
    ax[0,0].set_xlabel("No. Hold-out Data")
    handles, labels = ax[0,0].get_legend_handles_labels()

    # first row
    leg1 = ax[0,0].legend(
        handles[:5], labels[:5],
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        frameon=False
    )
    ax[0,0].add_artist(leg1)
    # second row
    ax[0,0].legend(
        handles[5:], labels[5:],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        frameon=False
    )
    fig.savefig("plots/learning_curve_uma_features.pdf")

if __name__ == '__main__':
    data = {
        0: {
            'dir': 'models/uma/learning_curve/',
            'identifier': None,
            'exclusions': None,
        },
    }
    columns = [
        'stats_type',
        'seed',
        'data.target',
        'data.cal_subset_size',
        'method.name',
        'model.variance_head.feature_keys',
        'model.variance_head._target_',
    ]
    metrics = [
        'mae',
        'lpd',
        'gaussian_nll',
        'ece',
        'ece_mixture',
    ]
    val_metric = 'gaussian_nll'
    methods = [
        'homo_nll',
        'post_hoc',
    ]
    target = 0
    df = generate_df(data, columns, metrics, val_metric, methods, target)
    gen_individual_plots(df)
