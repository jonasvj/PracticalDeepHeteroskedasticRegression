import math
import pandas as pd
from typing import Iterable

COL_TO_LABEL = {
    'data.target': 'Target',
    'method.name': 'Method',
    'gaussian_nll': r'$\textrm{NLL}\downarrow$',
    'lpd': r'$\textrm{NLL}\downarrow$',
    'ece': r'$\textrm{ECE}\downarrow$',
    'mae': r'$\textrm{MAE}\downarrow$',
    'method.calibrate.optimizer.weight_decay': r'$\lambda$',
}

FEATURE_IDX_TO_LABEL = {i: rf'$z^{i}$' for i in range(100)}

COLOR_MAP_TABLEAU_20 = {
    "0": 6,
    "1": 5,
    "2": 4,
    "3": 3,
    "4": 2,
    "5": 1, # This is all layers learnt jointly.
    "6": 0, # This is ensemble.
    "post_hoc": 0, # Same color as ensemble.
    "beta_nll": 10,
    "faithful": 12,
    "homo_nll": 14,
    "naive_nll": 16,
    "natural_nll": 18,
}

COLOR_MAP_TABLEAU_20_UMA = {
    "0": 8,
    "1": 7,
    "2": 6,
    "3": 5,
    "4": 4,
    "5": 3,
    "6": 2,
    "7": 1, # This is all layers learnt jointly.
    "8": 0, # This is ensemble.
    "post_hoc": 0, # Same color as ensemble.
}

COLOR_MAP_TABLEAU_20_ALLSCAIP = {
    "0": 9,
    "1": 8,
    "2": 7,
    "3": 6,
    "4": 5,
    "5": 4,
    "6": 3,
    "7": 2,
    "8": 1, # This is all layers learnt jointly.
    "9": 0, # This is ensemble.
    "post_hoc": 0, # Same color as ensemble.
}

COLOR_MAP_NAME = "Tableau_20"

QM9_TARGET_TO_LABEL = {
    0: r'$\mu$',
    1: r'$\alpha$',
    2: r'$\epsilon_{\textrm{HOMO}}$',
    3: r'$\epsilon_{\textrm{LUMO}}$',
    4: r'$\Delta \epsilon$',
    5: r'$\langle R^2\rangle$',
    6: r'$\textrm{ZPVE}$',
    7: r'$U_0$',
    8: r'$U$',
    9: r'$H$',
    10: r'$G$',
    11: r'$c_{\textrm{v}}$',
}

METHOD_TO_LABEL_WITH_LINESPLIT = {
    'homo_nll': r'\makecell{$\textrm{Homo.}$\\$\textrm{NLL}$}',
    'naive_nll': r'\makecell{$\textrm{Naïve}$\\$\textrm{NLL}$}',
    'natural_nll': r'\makecell{$\textrm{Nat.}$\\$\textrm{NLL}$}',
    'beta_nll': r'$\beta\textrm{-NLL}$',
    'faithful': r'$\textrm{Faithful}$',
    'post_hoc': r'\makecell{$\textrm{Post hoc}$\\$\textrm{(ours)}$}',
}

METHOD_TO_LABEL = {
    'homo_nll': r'$\textrm{Homo. NLL}$',
    'naive_nll': r'$\textrm{Naïve NLL}$',
    'natural_nll': r'$\textrm{Nat. NLL}$',
    'beta_nll': r'$\beta\textrm{-NLL}$',
    'faithful': r'$\textrm{Faithful}$',
    'post_hoc': r'$\textrm{Post-hoc}$'+'\n'+r'$\textrm{Ensemble}$',
    'Ensemble': r'$\textrm{Ensemble}$',
    'All': r'$\textrm{All}$',
}

UMA_LAYER_TO_INTEGER = {
    "module.backbone.blocks.0":0,
    "module.backbone.blocks.1":1,
    "module.backbone.blocks.2":2,
    "module.backbone.blocks.3":3,
    "module.output_heads.energyandforcehead.head.energy_block.0":4,
    "module.output_heads.energyandforcehead.head.energy_block.2":5,
    "module.output_heads.energyandforcehead.head.energy_block.4":6,
}

ALLSCAIP_LAYER_TO_INTEGER={
    'module.backbone.transformer_blocks.0.node_ffn.ffn_norm':0,
    'module.backbone.transformer_blocks.1.node_ffn.ffn_norm':1,
    'module.backbone.transformer_blocks.2.node_ffn.ffn_norm':2,
    'module.backbone.transformer_blocks.3.node_ffn.ffn_norm':3,
    'module.backbone.transformer_blocks.4.node_ffn.ffn_norm':4,
    'module.backbone.transformer_blocks.5.node_ffn.ffn_norm':5,
    'module.output_heads.energyandforcehead.head.energy_ffn.0.0':6,
    'module.output_heads.energyandforcehead.head.energy_ffn.1.0':7,
}

PRETRAINED_TO_LABEL = {
    'uma': r'$\textrm{UMA}$',
    'allscaip': r'$\textrm{AllScAIP}$',
}

def map_feature_UMA(key_list):
    if not isinstance(key_list, Iterable):
        return -1
    if len(key_list) == 1:
        # Single feature
        return UMA_LAYER_TO_INTEGER[key_list[0]]
    else:
        # Combined feature
        return len(UMA_LAYER_TO_INTEGER.keys())

def map_feature_allscaip(key_list):
    if not isinstance(key_list, Iterable):
        return -1
    if len(key_list) == 1:
        # Single feature
        return ALLSCAIP_LAYER_TO_INTEGER[key_list[0]]
    else:
        # Combined feature
        return len(ALLSCAIP_LAYER_TO_INTEGER.keys())



def get_feature(idx_list):
    if not isinstance(idx_list, Iterable):
        return idx_list

    if len(idx_list) == 1:
        # Single feature
        return idx_list[0]
    else:
        # Combined feature
        return max(idx_list) + 1


def apply_bolding_style(text_val, is_best):
    if is_best:
        text_val = f"\\textbf{{{text_val}}}"

    return text_val

def get_decimal_places(val, sig_figs=3, max_decimals=3):
    if val == 0:
        return 2  # Default to 0.00 for zero

    # 1. Calculate magnitude (e.g., 123 -> 2, 0.01 -> -2)
    mag = math.floor(math.log10(abs(val)))

    # 2. Calculate decimals needed for N sig figs
    # Formula: decimals = sig_figs - 1 - magnitude
    # Example: 1.234 (mag 0) -> 3 - 1 - 0 = 2 decimals (1.23)
    decimals = sig_figs - 1 - mag

    # 3. Apply constraints
    # Cap at max_decimals (e.g., 3)
    decimals = min(decimals, max_decimals)
    # Ensure non-negative (no rounding to tens/hundreds, just integers)
    decimals = max(0, decimals)

    return int(decimals)


def format_mean_sem_cell(mean_val, sem_val):
    """
    Formats as: Mean (Uncertainty)
    Uncertainty is the SEM scaled to the last digit of Mean.
    """
    # 1. Determine precision based on Mean
    decimals = get_decimal_places(mean_val, sig_figs=3, max_decimals=3)

    # 2. Format the Mean
    mean_str = f"{mean_val:.{decimals}f}"

    # 3. Calculate the Uncertainty Integer
    # We want the uncertainty in units of the last displayed digit.
    # If decimals=2, we multiply by 100. If decimals=0, multiply by 1.
    ci_val = sem_val
    scale_factor = 10 ** decimals
    ci_int = int(round(ci_val * scale_factor))

    # 4. Return combined string
    return f"{mean_str} {{\\scriptsize({ci_int})}}"


def get_best_mean_mask(df_mean, maximize_cols):
    mask = pd.DataFrame(False, index=df_mean.index, columns=df_mean.columns)
    idx = pd.IndexSlice


    for metric in df_mean.columns.levels[0]:
        is_maximize = 'uparrow' in metric or metric in maximize_cols
        means = df_mean.loc[:, idx[metric, :]]

        for row_idx in means.index:
            row_mean = means.loc[row_idx]

            # Skip rows where all values are NaN
            if row_mean.isna().all():
                continue

            # Find best results
            best_col = row_mean.idxmax() if is_maximize else row_mean.idxmin()

            # Set cell to be bolded
            mask.loc[row_idx, best_col] = True

    return mask
