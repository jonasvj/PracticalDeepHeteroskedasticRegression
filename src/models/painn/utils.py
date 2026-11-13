import torch.nn as nn


def build_readout_network(
    num_in_features: int,
    num_out_features: int = 1,
    num_layers: int = 2,
    activation: nn.Module = nn.SiLU,
    dropout_rate: float = 0.,
):
    """
    Build readout network.

    Args:
        num_in_features: Number of input features.
        num_out_features: Number of output features (targets).
        num_layers: Number of layers in the network.
        activation: Activation function as a nn.Module.
    
    Returns:
        The readout network as a nn.Module.
    """
    # Number of neurons in each layer
    num_neurons = [
        num_in_features,
        *[
            max(num_out_features, num_in_features // 2**(i + 1))
            for i in range(num_layers-1)
        ],
        num_out_features,
    ]

    # Build network
    readout_network = nn.Sequential()
    for i, (n_in, n_out) in enumerate(zip(num_neurons[:-1], num_neurons[1:])):
        readout_network.append(nn.Linear(n_in, n_out))
        if i < num_layers - 1:
            readout_network.append(activation())
            readout_network.append(nn.Dropout(p=dropout_rate))

    return readout_network