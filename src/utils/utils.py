import os
import torch
from typing import List, Optional
from collections.abc import MutableMapping
from pytorch_lightning import seed_everything


EPSILON = 1e-8


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    torch.backends.cudnn.benchmark = False      # Ensures algorithm selection is deterministic
    torch.backends.cudnn.deterministic = True   # Ensures the algorithm itself is deterministic
    torch.use_deterministic_algorithms(True, warn_only=True)

    seed_everything(seed, workers=True)


def stable_softplus(x):
    return torch.nn.functional.softplus(x) + EPSILON


def inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def inverse_stable_softplus(x):
    return inverse_softplus(x - EPSILON)


def flatten(dictionary, parent_key='', separator='.'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def parse_dirs(
    directories: List[str],
    identifier: Optional[str] = None,
    exclusions: Optional[List[str]] = None,
):
    all_results = list()
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if '.pt' in file:
                    full_path = os.path.join(root, file)
                    if exclusions is not None and any(exclusion in full_path for exclusion in exclusions):
                        continue
                    if identifier is None or identifier in full_path:
                        results = get_results(os.path.join(root, file))
                        all_results.extend(results)
    return all_results


def get_results(model_file: str):
    model_dict = torch.load(model_file, weights_only=False)
    cfg = model_dict['cfg']
    stats = model_dict['stats']
    preds = model_dict['preds']

    if 'feature-ensemble' not in model_file:
        stats_suffix = '_calibrated'
    else:
        stats_suffix = ''

    val_stats = stats['val' + stats_suffix]
    val_stats['stats_type'] = 'val'
    dict_val = {**flatten(cfg), **val_stats}

    test_stats = stats['test' + stats_suffix]
    test_stats['stats_type'] = 'test'
    test_stats['means_test_calibrated'] = preds['means_test'+stats_suffix]
    test_stats['vars_test_calibrated'] = preds['vars_test'+stats_suffix]
    test_stats['targets_test_calibrated'] = preds['targets_test'+stats_suffix]

    dict_test = {**flatten(cfg), **test_stats}

    results = [dict_val, dict_test]

    return results


def transform_shapes_for_variance_ensemble(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    assert mean.dim() == 1 and var.dim() == 2
    assert mean.shape[0] == var.shape[0]
    S = var.shape[1]
    mean = mean.unsqueeze(1).repeat(1, S)                           # N x S
    output = torch.stack([mean, var], dim=2)                        # N x S x K
    output = torch.transpose(output, 0, 1)                          # S x N x K

    return output
