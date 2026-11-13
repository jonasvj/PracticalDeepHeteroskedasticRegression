from torch.utils.data import DataLoader
from src.methods.base import RegressionBase
from src.utils.utils import set_seed


def evaluate(
    trainer: RegressionBase,
    dataloader: DataLoader,
    use_var_scale_factor: bool,
    name: str
):
    print(f'Computing {name} statistics.')
    set_seed(0)
    means, vars, targets = trainer.predict(
        dataloader=dataloader, use_var_scale_factor=use_var_scale_factor,
    )
    stats = trainer.compute_regression_stats(
        means, vars, targets
    )
    print(f'{name} statistics:')
    print(stats)
    print(f'Computing {name} statistics DONE.')
    return means, vars, targets, stats
