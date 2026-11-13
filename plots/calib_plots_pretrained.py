import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import List
from src.utils import parse_dirs
from tueplots import bundles
import seaborn as sns
import matplotlib as mlp
import matplotlib.pyplot as plt
from pypalettes import load_palette
from src.utils.plot_utils import (
    COL_TO_LABEL, COLOR_MAP_TABLEAU_20_UMA, COLOR_MAP_TABLEAU_20_ALLSCAIP,
    COLOR_MAP_NAME, map_feature_UMA, map_feature_allscaip
)


def main(
    data: dict,
    columns: List[str],
    metrics: List[str],
    val_metric: str,
    methods: List[str],
    model: str,
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

    # Remove all other than post-hoc methods
    df = df[df['method.name'] == "post_hoc"]

    if model == "uma":
        feature_map = map_feature_UMA
    elif model == "allscaip":
        feature_map = map_feature_allscaip

    # Simplify feature variable.
    df['feature'] = df['model.variance_head.feature_keys'].map(feature_map)


    # Set feature for ensemble.
    df.loc[
        df['model.variance_head._target_'] == 'src.models.FeatureModelEnsembleVariancePrediction', 
        'feature'
    ] = df['feature'].max(axis=None) + 1

    df['feature'] = pd.Categorical(
        df['feature'], df['feature'].unique().sort()
    )


    # Keep only test results
    df = df[df['stats_type'] == 'test']
    df = df.drop('stats_type', axis=1)
    
    return df
    
def get_single_run_curves(means, variances, targets):
    """
    Computes the raw x (fraction removed) and y (MAE) for Model and Oracle 
    for a single seed.
    """
    means = np.array(means)
    std_devs = np.sqrt(np.array(variances))
    targets = np.array(targets)
    abs_errors = np.abs(targets - means)
    N = len(targets)
    
    # --- Helper Calculation ---
    def calc_curve(errors, sort_metric):
        sorted_idx = np.argsort(sort_metric)[::-1] # Sort high to low
        sorted_errors = errors[sorted_idx]
        
        # Total sum of errors
        total_sum = np.sum(sorted_errors)
        # Errors removed so far
        cum_removed = np.cumsum(sorted_errors)
        
        # Remaining sum = Total - Removed
        remaining_sum = total_sum - cum_removed
        # Insert initial state (0 removed)
        remaining_sum = np.insert(remaining_sum, 0, total_sum)
        # Remove last point (0 remaining)
        remaining_sum = remaining_sum[:-1]
        
        remaining_counts = np.arange(N, 0, -1)
        return remaining_sum / remaining_counts

    # 1. Model Curve
    model_y = calc_curve(abs_errors, std_devs)
    
    # 2. Oracle Curve
    oracle_y = calc_curve(abs_errors, abs_errors)
    
    # 3. X-Axis
    x_axis = np.arange(0, N) / N
    
    return x_axis, model_y, oracle_y


def save_feature_plot(df, feature_id, model):
    """
    Re-runs the plotting logic for a SINGLE feature to save it as a high-quality
    standalone figure (e.g. for a presentation or detailed view).
    """
    # 1. Setup Style for Single Plot (Half Column)
    plt.rcParams.update(bundles.neurips2024(rel_width=0.25, nrows=1, ncols=1, usetex=True))
    
    # Select Color Map
    if model == "uma":
        COLOR_MAP = COLOR_MAP_TABLEAU_20_UMA
    elif model == "allscaip":
        COLOR_MAP = COLOR_MAP_TABLEAU_20_ALLSCAIP
    
    plt.rcParams['axes.prop_cycle'] = mlp.cycler(color=load_palette(COLOR_MAP_NAME))

    # 2. Filter Data
    unique_targets = df['data.target'].unique()
    current_target = unique_targets[0]
    
    subset = df[(df['feature'] == feature_id) & (df['data.target'] == current_target)]
    
    if subset.empty:
        print(f"Skipping feature {feature_id}: No data found.")
        return

    # 3. Calculate Curves
    # (Identify ensemble index again to handle variance shape correctly)
    all_features = sorted(df['feature'].unique())
    ensemble_idx = np.max(all_features)

    common_x = np.linspace(0, 0.99, 100)
    model_curves, oracle_curves = [], []

    for _, row in subset.iterrows():
        m = row['means_test_calibrated'].numpy() * 1000
        t = row['targets_test_calibrated'].numpy() * 1000
        
        # Ensemble check
        if feature_id == ensemble_idx:
            v = row['vars_test_calibrated'].numpy().mean(axis=-1)
        else:
            v = row['vars_test_calibrated'].numpy()
            
        raw_x, raw_model, raw_oracle = get_single_run_curves(m, v, t)
        
        f_model = interp1d(raw_x, raw_model, bounds_error=False, fill_value=(raw_model[0], raw_model[-1]))
        f_oracle = interp1d(raw_x, raw_oracle, bounds_error=False, fill_value=(raw_oracle[0], raw_oracle[-1]))
        
        model_curves.append(f_model(common_x))
        oracle_curves.append(f_oracle(common_x))

    # 4. Plot
    fig, ax = plt.subplots(figsize=(3.25/2, (3.25/2)*0.6180339887498949)) # Size handled by tueplots 'half'
    
    if model_curves:
        mu_model = np.nanmean(model_curves, axis=0)
        mu_oracle = np.nanmean(oracle_curves, axis=0)
        se_model = np.nanstd(model_curves, axis=0, ddof=1) / np.sqrt(len(model_curves))
        se_oracle = np.nanstd(oracle_curves, axis=0, ddof=1) / np.sqrt(len(oracle_curves))

        # Oracle
        ax.plot(common_x, mu_oracle, color='black', linestyle='--', alpha=0.8, label='Oracle')
        ax.fill_between(common_x, mu_oracle - se_oracle, mu_oracle + se_oracle, color='black', alpha=0.1)

        # Model
        feat_color = f'C{COLOR_MAP.get(str(feature_id), 0)}'
        ax.plot(common_x, mu_model, color=feat_color, linewidth=2, label='Model')
        ax.fill_between(common_x, mu_model - se_model, mu_model + se_model, color=feat_color, alpha=0.2)

    # 5. Formatting
    # Use the nice LaTeX label
    ax.set_xlabel("Frac. Removed")
    ax.set_ylabel(COL_TO_LABEL.get('mae', 'MAE'))
    ax.grid(True, linestyle=':', alpha=0.4)
    sns.despine(bottom=False, top=True, right=True, left=False)
    
    # Save
    fig.savefig(f"plots/calib_plot_pretrained_{model}_ensemble.pdf")


if __name__ == '__main__':
    columns = [
        'stats_type',
        'seed',
        'data.target',
        'method.name',
        'model.variance_head.feature_keys',
        'model.variance_head._target_',
    ]
    metrics = [
        'means_test_calibrated',
        'vars_test_calibrated',
        'targets_test_calibrated',
    ]
    val_metric = 'gaussian_nll'
    methods = [
        'post_hoc',
    ]

    for model_target in ['uma', 'allscaip']:
        data = {
            0: {
                'dir': f'models/{model_target}/post_hoc',
                'identifier': None,
                'exclusions': None,
            }
        }
        df = main(data, columns, metrics, val_metric, methods, model=model_target)

        ensemble_idx = np.max(sorted(df['feature'].unique()))
        save_feature_plot(df, feature_id=ensemble_idx, model=model_target)
