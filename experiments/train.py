import os
import re
import torch
import hydra
from src.utils import *
from src.methods import *
from copy import deepcopy
from omegaconf import ListConfig


@hydra.main(
    config_path='conf/',
    config_name='config.yaml',
    version_base=None,
)
def main(cfg):
    set_seed(cfg.seed)
    print(cfg)
    if not os.path.isdir(cfg.model_dir):
        os.makedirs(cfg.model_dir)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('Initializing data.')
    dm = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()
    dm.setup()
    print('Initializing data DONE.')

    # Model
    model = init_model(cfg, dm)

    # Possibly load pretrained model
    if cfg.pretrained_model is not None:
        print('Loading pre-trained model.')
        if not isinstance(cfg.pretrained_model, ListConfig):
            state_dict = torch.load(cfg.pretrained_model, weights_only=False)['state_dict']
            if 'variance_head.var_scale_factor' in state_dict:
                del state_dict['variance_head.var_scale_factor']
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict,
                strict=False
            )
        # Load ensemble of variance heads
        else:
            state_dicts = [
                torch.load(dir_, weights_only=False)['state_dict'] for dir_ in cfg.pretrained_model
            ]
            new_state_dict = deepcopy(state_dicts[0])

            # Delete variance head from new state dict
            for key in list(new_state_dict.keys()):
                if 'variance_head' in key:
                    del new_state_dict[key]

            # Insert ensemble of variance heads
            for key in list(model.state_dict().keys()):
                if 'variance_heads' in key:
                    lkey, rkey = re.split(r'.\d.', key)
                    k = int(key.split(lkey + '.')[-1].split('.' + rkey)[0])

                    k_key = key.replace(f'variance_heads.{k}.', '')
                    new_state_dict[key] = state_dicts[k][k_key]

            missing_keys, unexpected_keys = model.load_state_dict(
                new_state_dict,
                strict=False
            )

        print('Missing keys:', missing_keys)
        print('Unexpected keys:', unexpected_keys)
        assert all('variance_head' in key for key in missing_keys)
        print('Loading pre-trained model DONE.')
    model.to(device)

    # Initialize regression method
    print('Initializing regression.')
    trainer = hydra.utils.instantiate(cfg.method.init, model=model)

    # Fit model with inference method
    if cfg.do_training:
        print('Fitting model.')
        hydra.utils.call(
            cfg.method.fit,
            trainer,
            train_loader=dm.train_dataloader(),
            val_loader=dm.val_dataloader(),
        )
        print('Fitting model DONE.')

    stats = {}
    preds = {}

    # Val stats
    _, _, _, val_stats = evaluate(
        trainer, dm.val_dataloader(), False, 'val'
    )
    stats['val'] = val_stats

    # Test stats
    means_test, vars_test, targets_test, test_stats = evaluate(
        trainer, dm.test_dataloader(), cfg.use_var_scale_during_normal_test, 'test'
    )
    stats['test'] = test_stats
    preds['means_test'] = means_test
    preds['vars_test'] = vars_test
    preds['targets_test'] = targets_test

    if cfg.do_calibration:
        print('Calibrating model.')
        hydra.utils.call(
            cfg.method.calibrate,
            trainer,
            cal_loader=dm.cal_dataloader(),
            val_loader=dm.val_dataloader(),
        )
        print('Calibrating model DONE.')

        # Val stats
        _, _, _, val_stats = evaluate(
            trainer, dm.val_dataloader(), True, 'val_calibrated'
        )
        stats['val_calibrated'] = val_stats

        # Test stats
        means_test, vars_test, targets_test, test_stats = evaluate(
            trainer, dm.test_dataloader(), True, 'test_calibrated'
        )
        stats['test_calibrated'] = test_stats
        preds['means_test_calibrated'] = means_test
        preds['vars_test_calibrated'] = vars_test
        preds['targets_test_calibrated'] = targets_test

    # Save model
    print('Saving model.')
    model = trainer.model
    model.to('cpu')
    torch.save(
        {'cfg': cfg, 'state_dict': model.state_dict(), 'stats': stats, 'preds': preds},
        os.path.join(cfg.model_dir, cfg.model_name).lower()
    )
    print('Saving model DONE.')


if __name__ == '__main__':
    main()
