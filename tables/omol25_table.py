import pandas as pd
from typing import List
from src.utils import parse_dirs
from src.utils.plot_utils import (
    format_mean_sem_cell, apply_bolding_style,
    COL_TO_LABEL, METHOD_TO_LABEL, PRETRAINED_TO_LABEL,
)


def generate_df(models: List[str]):
    dfs = []
    for model in models:
        results = parse_dirs(
            directories=[f'models/{model}'],
            exclusions=['learning_curve'],
        )
        df = pd.DataFrame(results)
        df['model'] = model
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    df = df[df['stats_type'] == 'test']

    is_ensemble = df['model.variance_head._target_'] == 'src.models.FeatureModelEnsembleVariancePrediction'
    is_all = df['model_dir'].str.endswith('feature-all')
    df.loc[is_ensemble, 'method.name'] = 'Ensemble'
    df.loc[is_all, 'method.name'] = 'All'

    # Use the NLL and ECE of the Gaussian mixture for the ensemble
    df.loc[is_ensemble, 'gaussian_nll'] = -df.loc[is_ensemble, 'lpd']
    df.loc[is_ensemble, 'ece'] = df.loc[is_ensemble, 'ece_mixture']

    df['mae'] = 1000*df['mae']

    return df


def gen_table(df, models: List[str], methods: List[str], table_name: str):
    df = df[df['method.name'].isin(methods)]
    counts = df.groupby(['method.name', 'model'])['mae'].count()
    assert (counts == 3).all(), f'Expected 3 seeds per group, got:\n{counts}'

    metrics = ['mae', 'gaussian_nll', 'ece']
    mean = df.groupby(['method.name', 'model'])[metrics].mean()
    sem = df.groupby(['method.name', 'model'])[metrics].sem()

    def cell(method, model, metric, is_best=False):
        text = format_mean_sem_cell(mean.loc[(method, model), metric], sem.loc[(method, model), metric])
        return apply_bolding_style(text, is_best)

    # All methods share the mean model, so the MAE is only reported once
    lines = [
        r'\begin{tabular}{llcc}',
        r'\toprule',
        r'\multicolumn{2}{l}{} & \multicolumn{2}{c}{$\textrm{Model}$} \\',
        r'\cmidrule(lr){3-4}',
        r'\multicolumn{2}{l}{$\textrm{Metric}$} & ' + ' & '.join(PRETRAINED_TO_LABEL[m] for m in models) + r' \\',
        r'\midrule',
        rf'\multicolumn{{2}}{{l}}{{{COL_TO_LABEL["mae"]}}} & ' + ' & '.join(cell('homo_nll', m, 'mae') for m in models) + r' \\',
    ]
    for metric in ['gaussian_nll', 'ece']:
        lines.append(r'\cmidrule(lr){1-4}')
        best = {m: mean.xs(m, level='model')[metric].idxmin() for m in models}
        for i, method in enumerate(methods):
            label = rf'\multirow{{{len(methods)}}}{{*}}{{{COL_TO_LABEL[metric]}}}' if i == 0 else ''
            cells = [cell(method, m, metric, best[m] == method) for m in models]
            lines.append(f'{label} & {METHOD_TO_LABEL[method]} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']

    with open(table_name, 'w') as f:
        print('\n'.join(lines), file=f)


if __name__ == '__main__':
    models = ['uma', 'allscaip']
    methods = ['homo_nll', 'All', 'Ensemble']
    table_name = 'tables/omol25_table.tex'
    df = generate_df(models)
    gen_table(df, models, methods, table_name)
