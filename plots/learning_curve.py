import pandas as pd
from typing import List
from src.utils import parse_dirs
from tueplots import bundles
import matplotlib.pyplot as plt
import matplotlib as mlp
from pypalettes import load_palette
from src.utils.plot_utils import (
    get_feature, COL_TO_LABEL, FEATURE_IDX_TO_LABEL, COLOR_MAP_TABLEAU_20,
    COLOR_MAP_NAME, METHOD_TO_LABEL,
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
    df = df[columns + list(set([*metrics, val_metric]))]
    df = df[df['method.name'].isin(methods)]
    
        
    # Keep only selected target
    df = df[df['data.target'] == target]

    # Select best beta value
    df_beta = df[df['method.name'] == 'beta_nll']

    # Merge test and val stats
    df_beta = df_beta[df_beta['stats_type'] == 'test'].merge(
        df_beta[df_beta['stats_type'] == 'val'],
        on=['seed', 'data.target', 'method.init.beta', 'data.cal_subset_size'],
        suffixes=(None, '_val'),
    )
    # Select betas with best val NLL for each seed and target
    idx = df_beta.groupby(['seed', 'data.target', 'data.cal_subset_size'])[val_metric + '_val'].idxmin()
    df_beta = df_beta.loc[idx]

    # Remove validation data
    df_beta = df_beta[[col for col in df_beta.columns if '_val' not in col]]

    # Merge with original results
    df = df[df['method.name'] != 'beta_nll']
    df = pd.concat([df, df_beta])
    df = df.drop('method.init.beta', axis=1)

    # Keep only test results
    df = df[df['stats_type'] == 'test']
    df = df.drop('stats_type', axis=1)

    # Simplify feature variable.
    df['feature'] = df['model.feature_extractor.module_indices_to_extract'].map(get_feature)
    df = df.drop('model.feature_extractor.module_indices_to_extract', axis=1)

    # Set feature for ensemble.
    df.loc[
        df['model.variance_head._target_'] == 'src.models.painn.PainNNEnsembleVariancePrediction', 
        'feature'
    ] = df['feature'].max(axis=None) + 1
    df = df.drop('model.variance_head._target_', axis=1)

    df['feature'] = pd.Categorical(
        df['feature'], df['feature'].unique().sort()
    )

    features = df['feature'].unique()
    
    # Convert NLL and ECE of ensemble to -lpd and ece_mixture for specifically ensemble
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
    # first row (5 items)
    leg1 = ax[0,0].legend(
        handles[:3], labels[:3],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        frameon=False
    )
    ax[0,0].add_artist(leg1)
    # second row (2 items)
    ax[0,0].legend(
        handles[3:], labels[3:],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.45),
        frameon=False
    )
    ax[0,0].add_artist(leg1)
    fig.savefig("plots/learning_curve_qm9_methods.pdf")

    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)

    features = [0,1,2,3,4,5,6]
    for feature in features:
        mean_tmp = mean[(mean['feature'] == feature) & (mean['method.name'] == "post_hoc")].sort_values(by=x_axis)
        sem_tmp = sem[(sem['feature'] == feature) & (sem['method.name'] == "post_hoc")].sort_values(by=x_axis)
        ax[0,0].errorbar(
            mean_tmp[x_axis]*num_cal_points_full,
            mean_tmp[metric_to_plot],
            yerr=sem_tmp[metric_to_plot],
            label=FEATURE_IDX_TO_LABEL[feature],
            color=f'C{COLOR_MAP_TABLEAU_20[str(feature)]}',
        )

    ax[0,0].spines['top'].set_visible(False)
    ax[0,0].spines['right'].set_visible(False)
    ax[0,0].grid(True, zorder=-1, color='grey', linewidth=0.5)
    ax[0,0].set_axisbelow(True)
    ax[0,0].set_xscale('log', base=10)
    ax[0,0].set_ylim([None, 0])

    ax[0,0].set_ylabel(COL_TO_LABEL[metric_to_plot])
    ax[0,0].set_xlabel("No. Hold-out Data")
    handles, labels = ax[0,0].get_legend_handles_labels()

    # first row (5 items)
    leg1 = ax[0,0].legend(
        handles[:4], labels[:4],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        frameon=False
    )
    ax[0,0].add_artist(leg1)
    # second row (2 items)
    ax[0,0].legend(
        handles[4:], labels[4:],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.45),
        frameon=False
    )
    fig.savefig("plots/learning_curve_qm9_features.pdf", bbox_inches='tight')

if __name__ == '__main__':
    data = {
        0: {
            'dir': 'models/qm9/learning_curve',
            'identifier': None,
            'exclusions': []
        },
    }
    columns = [
        'stats_type',
        'seed',
        'data.target',
        'data.cal_subset_size',
        'method.name',
        'method.init.beta',
        'model.feature_extractor.module_indices_to_extract',
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
        'naive_nll',
        'natural_nll',
        'beta_nll',
        'faithful',
        'post_hoc',
    ]
    target = 8
    df = generate_df(data, columns, metrics, val_metric, methods, target)

    # SHOULD BE 288 ROWS (SEEDS=3, METHODS=(5+7), SUBSET_SIZES=8, 8*3*12)
    print(df.shape[0])
    assert df.shape[0] == 288

    # ASSERT THAT LPD AND ECE HAVE BEEN SET PROPERLY FOR ENSEMBLE RUNS
    run_with_ensemble = df[df['feature'] == 6].iloc[0]
    assert -run_with_ensemble['lpd'] == run_with_ensemble['gaussian_nll']
    assert run_with_ensemble['ece'] == run_with_ensemble['ece_mixture']

    gen_individual_plots(df)