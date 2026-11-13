import hydra


def init_model(cfg, dm):
    """
    Build a model from a hydra config, injecting the target statistics that the
    prediction heads need.
    """
    print('Initializing model.')
    if cfg.model.name == 'painn':
        y_mean, y_std, atom_refs = dm.get_target_stats(
            remove_atom_refs=True,
            divide_by_atoms=True,
        )
        head = hydra.utils.instantiate(cfg.model.head)
        head = head(mean=y_mean.item(), std=y_std.item(), atom_refs=atom_refs)
        model = hydra.utils.instantiate(cfg.model, head=head)
    elif cfg.model.name in ('resnet', 'ft_transformer'):
        print(f'Configuring {cfg.model.name} model with target stats.')
        y_mean, y_std = dm.get_target_stats()
        head = hydra.utils.instantiate(cfg.model.head)
        head = head(mean=y_mean.item(), std=y_std.item())
        model = hydra.utils.instantiate(cfg.model, head=head)
    else:
        model = hydra.utils.instantiate(cfg.model)
    print('Initializing model DONE.')

    return model
