import pandas as pd
from typing import List
from src.utils import parse_dirs
from src.data import QM9DataModule
from src.utils.plot_utils import (
    get_feature, get_best_mean_mask, format_mean_sem_cell,
    COL_TO_LABEL, QM9_TARGET_TO_LABEL,
    METHOD_TO_LABEL_WITH_LINESPLIT, apply_bolding_style,
)

DATASET_TO_LABEL = {
    'california_housing': r'Cal. Housing',
    'year': r'Year',
}


def _select_best_beta(df, val_metric, groupby_keys):
    """Select best beta for beta_nll by val metric per (seed, *groupby_keys)."""
    df_beta = df[df['method.name'] == 'beta_nll']
    df_beta = df_beta[df_beta['stats_type'] == 'test'].merge(
        df_beta[df_beta['stats_type'] == 'val'],
        on=['seed', *groupby_keys, 'method.init.beta'],
        suffixes=(None, '_val'),
    )
    idx = df_beta.groupby(['seed', *groupby_keys])[val_metric + '_val'].idxmin()
    df_beta = df_beta.loc[idx]
    return df_beta[[col for col in df_beta.columns if '_val' not in col]]


def _preprocess_dataset(df, metrics, val_metric, methods, expected_seeds):
    """
    Common preprocessing pipeline shared across all datasets:
    - Filter to requested methods and available metric columns
    - Convert lpd -> gaussian_nll for post_hoc
    - Assert seed count per (data.target, method, beta) group
    - Select best beta_nll via val metric
    - Filter to test set and max-feature post_hoc ensemble
    Returns processed DataFrame with seed column retained.
    """
    keep_cols = (
        ['stats_type', 'seed', 'data.target', 'method.name', 'method.init.beta',
         'model.feature_extractor.module_indices_to_extract']
        + list(set([*(m for m in metrics if m in df.columns), val_metric]))
    )
    df = df[[col for col in df.columns if col in keep_cols]]
    df = df[df['method.name'].isin(methods)]

    if 'lpd' in df.columns and 'gaussian_nll' in df.columns:
        df.loc[df['method.name'] == 'post_hoc', 'gaussian_nll'] = (
            -df.loc[df['method.name'] == 'post_hoc', 'lpd']
        )
        df = df.drop('lpd', axis=1)

    # Check that all results are present
    counts = df[df['stats_type'] == 'test'].groupby(
        ['data.target', 'method.name', 'method.init.beta'],
        dropna=False,
    )[val_metric].count()
    with pd.option_context("display.max_rows", None):
        print(counts)
    assert (counts == expected_seeds).all(), (
        f"Expected {expected_seeds} seeds per group:\n{counts[counts != expected_seeds]}"
    )

    # Select best beta value
    df_beta = _select_best_beta(df, val_metric, ['data.target'])
    df = df[df['method.name'] != 'beta_nll']
    df = pd.concat([df, df_beta])
    df = df.drop('method.init.beta', axis=1)

    # Keep only test results
    df = df[df['stats_type'] == 'test'].drop('stats_type', axis=1)

    # Simplify feature variable and keep only max-feature ensemble for post_hoc
    df['feature'] = df['model.feature_extractor.module_indices_to_extract'].map(get_feature)
    df = df.drop('model.feature_extractor.module_indices_to_extract', axis=1)
    df = df[(df['method.name'] != 'post_hoc') | (df['feature'] == df['feature'].max())]
    df = df.drop('feature', axis=1)

    df['method.name'] = pd.Categorical(df['method.name'], methods)
    return df


def main(
    qm9_data: dict,
    tabular_datasets: List[str],
    metrics: List[str],
    val_metric: str,
    methods: List[str],
    table_name: str,
):
    all_dfs = []

    # --- QM9 (3 seeds) ---
    qm9_results = []
    for value in qm9_data.values():
        qm9_results.extend(
            parse_dirs(
                directories=[value['dir']],
                identifier=value['identifier'],
                exclusions=value['exclusions'],
            )
        )
    df_qm9 = _preprocess_dataset(
        pd.DataFrame(qm9_results), metrics, val_metric, methods, expected_seeds=3,
    )
    # Convert MAE units for QM9
    if 'mae' in df_qm9.columns:
        for idx, func in QM9DataModule.unit_conversion.items():
            if idx in df_qm9['data.target'].unique():
                df_qm9.loc[df_qm9['data.target'] == idx, 'mae'] = (
                    func(df_qm9[df_qm9['data.target'] == idx]['mae'])
                )
    # Preserve QM9 integer order before mapping to string labels
    qm9_target_order = [QM9_TARGET_TO_LABEL[i] for i in sorted(QM9_TARGET_TO_LABEL) if i in df_qm9['data.target'].unique()]
    df_qm9['data.target'] = pd.Categorical(
        df_qm9['data.target'].map(QM9_TARGET_TO_LABEL).fillna(df_qm9['data.target']),
        categories=qm9_target_order,
        ordered=True,
    )
    n_qm9 = df_qm9['data.target'].nunique()
    all_dfs.append(df_qm9)

    # --- UTKFace (3 seeds) ---
    utkface_baseline = parse_dirs(
        directories=['models/utkface'],
        identifier=None,
        exclusions=['post_hoc'],
    )
    utkface_ensemble = parse_dirs(
        directories=['models/utkface/post_hoc'],
        identifier='feature-ensemble',
        exclusions=None,
    )
    df_utk = _preprocess_dataset(
        pd.concat([pd.DataFrame(utkface_baseline), pd.DataFrame(utkface_ensemble)], ignore_index=True),
        metrics, val_metric, methods, expected_seeds=3,
    )
    df_utk['data.target'] = df_utk['data.target'].map({'age': 'UTKFace'}).fillna(df_utk['data.target'])
    n_utk = df_utk['data.target'].nunique()
    all_dfs.append(df_utk)

    # --- Tabular datasets (5 seeds) ---
    tabular_dfs = []
    for dataset in tabular_datasets:
        tab_baseline = parse_dirs(
            directories=[f'models/{dataset}'],
            identifier=None,
            exclusions=['post_hoc'],
        )
        tabular_dfs.append(pd.DataFrame(tab_baseline))
        tab_ensemble = parse_dirs(
            directories=[f'models/{dataset}'],
            identifier='feature-ensemble',
            exclusions=None,
        )
        if tab_ensemble:
            tabular_dfs.append(pd.DataFrame(tab_ensemble))
    df_tab = _preprocess_dataset(
        pd.concat(tabular_dfs, ignore_index=True),
        metrics, val_metric, methods, expected_seeds=5,
    )
    df_tab['data.target'] = df_tab['data.target'].map(DATASET_TO_LABEL).fillna(df_tab['data.target'])
    n_tab = df_tab['data.target'].nunique()
    all_dfs.append(df_tab)

    # --- Combine all datasets ---
    df = pd.concat(all_dfs, ignore_index=True)
    df['method.name'] = pd.Categorical(df['method.name'], methods)

    # Preserve dataset group order (QM9 → UTKFace → Tabular) from concatenation order.
    target_order = list(dict.fromkeys(
        str(t)
        for df_ in all_dfs
        for t in (df_['data.target'].cat.categories if hasattr(df_['data.target'], 'cat') else df_['data.target'].unique())
        if pd.notna(t)
    ))
    df['data.target'] = pd.Categorical(df['data.target'].astype(str), categories=target_order, ordered=True)

    # Final metric list (lpd already converted per dataset)
    final_metrics = [m for m in metrics if m != 'lpd' and m in df.columns]

    # Average over seeds
    df = df.drop('seed', axis=1)
    groupby_columns = list(set(df.columns) - set([*final_metrics, val_metric]))
    mean = df.groupby(groupby_columns, observed=True).mean().reset_index()
    sem = df.groupby(groupby_columns, observed=True).sem().reset_index()

    # Rename columns
    mean = mean.rename(COL_TO_LABEL, axis=1)
    sem = sem.rename(COL_TO_LABEL, axis=1)

    # Pivot tables for means and SEM
    table_mean = mean.pivot_table(
        index=['Target'],
        columns='Method',
        values=[COL_TO_LABEL[m] for m in final_metrics],
        sort=False,
        observed=True,
    )
    table_sem = sem.pivot_table(
        index=['Target'],
        columns='Method',
        values=[COL_TO_LABEL[m] for m in final_metrics],
        sort=False,
        observed=True,
    )

    # Enforce row order (pivot_table may re-sort the index)
    table_mean = table_mean.reindex(target_order)
    table_sem = table_sem.reindex(target_order)

    for tbl in [table_mean, table_sem]:
        tbl.rename(columns=METHOD_TO_LABEL_WITH_LINESPLIT, level=1, inplace=True)
        tbl.columns.names = [None, None]

    # Identify which metric columns are "maximize"
    maximize_cols = [c for c in table_mean.columns.levels[0] if 'uparrow' in c]
    best_mean_mask = get_best_mean_mask(table_mean, maximize_cols)

    # Build display table with formatted strings
    table_display = pd.DataFrame(index=table_mean.index, columns=table_mean.columns)
    for col in table_mean.columns:
        table_display[col] = [
            apply_bolding_style(format_mean_sem_cell(m, s), best)
            if pd.notna(m) else '---'
            for m, s, best in zip(table_mean[col], table_sem[col], best_mean_mask[col])
        ]

    # Export to LaTeX
    latex = table_display.to_latex(
        escape=False,
        multicolumn_format='c',
    )

    latex_list = latex.splitlines()

    # Cmidrule column numbers shifted +1 for the extra label column added below.
    cmidrules = ' '.join(
        rf'\cmidrule(lr){{{3 + i * len(methods)}-{2 + (i + 1) * len(methods)}}}'
        for i in range(len(final_metrics))
    )
    latex_list.insert(3, cmidrules)

    # After cmidrule insert, structure is:
    # [0] \begin{tabular}  [1] \toprule  [2] metric-group headers
    # [3] cmidrules        [4] method headers  [5] index-name row  [6] \midrule
    # [7 .. 7+n_qm9-1]         QM9 rows
    # [7+n_qm9 .. 7+n_qm9+n_utk-1]     UTKFace rows
    # [7+n_qm9+n_utk .. 7+n_qm9+n_utk+n_tab-1]  Tabular rows
    header_lines = 7
    total_cols = 1 + len(methods) * len(final_metrics)

    # Widen tabular spec for the extra label column.
    latex_list[0] = latex_list[0].replace(
        '{' + 'l' * total_cols + '}',
        '{l' + 'l' * total_cols + '}'
    )

    # Insert \midrule between groups (bottom-up to avoid index shifting).
    latex_list.insert(header_lines + n_qm9 + n_utk, r'\midrule')
    latex_list.insert(header_lines + n_qm9, r'\midrule')

    # After insertions:
    # QM9:     [header_lines .. header_lines+n_qm9-1]
    # \midrule: [header_lines+n_qm9]
    # UTKFace: [header_lines+n_qm9+1 .. header_lines+n_qm9+n_utk]
    # \midrule: [header_lines+n_qm9+n_utk+1]
    # Tabular: [header_lines+n_qm9+n_utk+2 .. header_lines+n_qm9+n_utk+n_tab+1]

    # Prepend extra empty cell to header lines.
    for i in [2, 4, 5]:
        latex_list[i] = ' & ' + latex_list[i]
    # Remove the auto-generated "Target" index name label.
    latex_list[5] = latex_list[5].replace('Target', '', 1)

    # First QM9 row gets the rotated multirow label.
    latex_list[header_lines] = (
        rf'\multirow{{{n_qm9}}}{{*}}{{\rotatebox{{90}}{{QM9}}}} & '
        + latex_list[header_lines]
    )
    # Remaining QM9 rows: empty label cell.
    for i in range(header_lines + 1, header_lines + n_qm9):
        latex_list[i] = ' & ' + latex_list[i]

    def _span_target(row):
        """Wrap the target name in \\multicolumn{2}{l}{...} so it spans both
        label columns (same width as QM9 rotated-label + target-name columns)."""
        parts = row.split(' & ')
        t = next(j for j, p in enumerate(parts) if p.strip())
        d = t + 1
        while d < len(parts) and not parts[d].strip():
            d += 1
        return rf'\multicolumn{{2}}{{l}}{{{parts[t].strip()}}} & ' + ' & '.join(parts[d:])

    # UTKFace rows: target name spans label cols 1-2.
    utk_start = header_lines + n_qm9 + 1
    for i in range(utk_start, utk_start + n_utk):
        latex_list[i] = _span_target(latex_list[i])

    # Tabular rows: same treatment.
    tab_start = header_lines + n_qm9 + n_utk + 2
    for i in range(tab_start, tab_start + n_tab):
        latex_list[i] = _span_target(latex_list[i])

    latex = '\n'.join(latex_list)
    with open(table_name, 'w') as f:
        print(latex, file=f)


if __name__ == '__main__':
    qm9_data = {
        0: {
            'dir': 'models/qm9',
            'identifier': 'feature-ensemble',
            'exclusions': ['models/qm9/wd_sweep', 'models/qm9/learning_curve'],
        },
        1: {
            'dir': 'models/qm9',
            'identifier': None,
            'exclusions': ['post_hoc', 'models/qm9/wd_sweep', 'models/qm9/learning_curve'],
        },
    }
    tabular_datasets = ['california_housing', 'year']
    metrics = [
        'mae',
        'lpd',
        'gaussian_nll',
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
    table_name = 'tables/baseline_table_combined.tex'
    main(qm9_data, tabular_datasets, metrics, val_metric, methods, table_name)
