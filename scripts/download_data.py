"""Download every dataset the way train.py does (instantiate the data config, prepare_data, setup)."""
import os
import hydra
from hydra import initialize_config_dir, compose

repo = os.getcwd()
with initialize_config_dir(config_dir=f'{repo}/experiments/conf', version_base=None):
    for name in ['qm9', 'utkface', 'california_housing', 'year']:
        cfg = compose(config_name='config', overrides=[f'data={name}'])
        dm = hydra.utils.instantiate(cfg.data)
        dm.prepare_data()
        dm.setup()
        print(f'{name}: train {len(dm.data_train)}, val {len(dm.data_val)}, test {len(dm.data_test)}', flush=True)
print('DOWNLOADS OK')
