import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.painn.utils import build_readout_network
from torch_scatter import segment_sum_csr
from typing import Optional, Union, TypedDict
from src.utils import stable_softplus, inverse_stable_softplus


class PredictionInput(TypedDict):
    z: Union[torch.IntTensor, torch.LongTensor]
    ptr: Union[torch.IntTensor, torch.LongTensor]
    scalar_features: torch.Tensor
    features: dict
    std: float
    use_var_scale_factor: bool


class PaiNNAtomwisePrediction(nn.Module):
    """
    Module for predicting properties as sums of atomic contributions. We 
    predict the property (energy) of a molecule as
        y = sum_i (E[o_i]*sigma + mu + e_i)
    where i indexes atoms, E[o_i] is the predicted mean for atom i in the 
    molecule, mu and sigma are the (per-atom) sample mean and std. deviation 
    of the property (energy), and e_i is the reference property (energy) for 
    atom i (based on its atom type).
    """
    def __init__(
        self, 
        num_features: int = 128,
        num_outputs: int = 1,
        num_layers: int = 2,
        mean: float = 0.,
        std: float = 1.,
        atom_refs: Optional[torch.Tensor] = None,
        dropout_rate: float = 0.,
    ) -> None:
        """
        Args:
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_outputs: Size of output, i.e., the number of targets/properties.
            num_layers: Number of layers in the fully-connected feedforward
                network.
            mean: torch.Tensor with mean value to shift atomwise contributions
                by.
            std: torch.Tensor with standard deviation to scale atomwise
                contributions by.
            atom_refs: torch.Tensor of size [num_atom_types, 1] with atomic
                reference values.
            dropout_rate: Dropout probability for dropout layers.
        """
        super().__init__()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.register_buffer(
            'mean',
            mean.clone().detach() if isinstance(mean, torch.Tensor) else torch.tensor(mean)
        )
        self.register_buffer(
            'std',
            std.clone().detach() if isinstance(std, torch.Tensor) else torch.tensor(std)
        )
        self.register_buffer('atom_refs', atom_refs)

        # Assume 1-dimensional outputs for now
        assert self.num_outputs == 1

        self.readout_network = build_readout_network(
            num_in_features=self.num_features,
            num_out_features=self.num_outputs,
            num_layers=self.num_layers,
            activation=nn.SiLU,
            dropout_rate=dropout_rate,
        )

        if atom_refs is not None:
            self.atom_refs = nn.Embedding.from_pretrained(atom_refs, freeze=True)
        else:
            self.atom_refs = None


    def forward(self, input_: PredictionInput) -> torch.Tensor:
        """
        Forward pass of atomwise readout network.

        Args:
            input_: Dictionary with the following key-value pairs:
                z: torch.Tensor of size [num_nodes] with atom type of each
                    node in the graph.
                ptr: torch.Tensor of size [num_graphs + 1] with the indexes
                    of the first node of each graph in the batch.
                scalar_features: torch.Tensor of size [num_nodes, num_features] 
                    with scalar features of each node.

        Returns:
            A tensor of size [num_graphs, num_outputs] with predictions for
            each graph.
        """
        atoms = input_['z']
        ptr = input_['ptr']
        scalar_features = input_['scalar_features']

        # Get atomwise contributions
        atomwise_contributions = self.readout_network(scalar_features)              # [num_nodes, num_outputs]

        # Scale and shift
        atomwise_contributions = self.std*atomwise_contributions + self.mean

        # Add atomic references
        if self.atom_refs is not None:
            atomwise_contributions = atomwise_contributions + self.atom_refs(atoms)

        # Sum contributions for each graph
        output_per_graph = segment_sum_csr(atomwise_contributions, ptr)

        return output_per_graph


class PaiNNLinearVariancePrediction(nn.Module):
    """
    Module for predicting property variances as sums of atomic contributions. 
    We predict the property (energy) variance of a molecule as
        y = sum_i (V[o_i]*sigma^2),
    where i indexes atoms, V[o_i] is the predicted variance for atom i in the 
    molecule, sigma is the (per-atom) sample std. deviation of the property 
    (energy). The variance V[o_i] is predicted as a linear combination of 
    features extrated from PaiNN, i.e.,
        V[o_i] = sum_j W_j z_{j,i}, 
    where z_{j_i} is the jth representation of atom i.
    """

    def __init__(self) -> None:
       super().__init__()
       self.linear = nn.LazyLinear(1)

       # Scale factor that can be fitted with calibration data
       self.var_scale_factor = nn.Parameter(
            inverse_stable_softplus(torch.tensor(1.0)),
            requires_grad=False
        )


    def forward(self, input_: PredictionInput) -> torch.Tensor:
        """
        Args:
            input_: Dictionary with the following key-value pairs:
                ptr: torch.Tensor of size [num_graphs + 1] with the indexes
                    of the first node of each graph in the batch.
                features: dictionary with features from different layers.
                std: float of per-atom data standard deviation.
                use_var_scale_factor: bool indicatinf wheter to scale variance 
                    with var_scale_factor.

        Returns:
            Variance per graph.
        """
        ptr = input_['ptr']
        features = input_['features']
        std = input_['std']
        use_var_scale_factor = input_['use_var_scale_factor']

        all_features = torch.cat([
            f[0] if isinstance(f, (tuple, list)) else f 
            for f in features.values()
        ], dim=-1)
        variance = stable_softplus(self.linear(all_features)) # [num_nodes, 1]
        variance_per_graph = segment_sum_csr(variance, ptr)   # [num_graphs, 1]

        scale = std**2
        if use_var_scale_factor:
            scale = scale*stable_softplus(self.var_scale_factor)

        return scale*variance_per_graph


class PaiNNMLPVariancePrediction(nn.Module):
    """
    Module for predicting property variances as sums of atomic contributions. 
    We predict the property (energy) variance of a molecule as
        y = sum_i (V[o_i]*sigma^2),
    where i indexes atoms, V[o_i] is the predicted variance for atom i in the 
    molecule, sigma is the (per-atom) sample std. deviation of the property 
    (energy). The variance V[o_i] is predicted with a one-hidden-layer MLP of
    features extracted from PaiNN.
    """

    def __init__(self, hidden_dim: int = 512) -> None:
       super().__init__()
       self.hidden_dim = hidden_dim
       self.linear_1 = nn.LazyLinear(self.hidden_dim)
       self.linear_2 = nn.Linear(in_features=self.hidden_dim, out_features=1)

       # Scale factor that can be fitted with calibration data
       self.var_scale_factor = nn.Parameter(
            inverse_stable_softplus(torch.tensor(1.0)),
            requires_grad=False
        )


    def forward(self, input_: PredictionInput) -> torch.Tensor:
        """
        Args:
            input_: Dictionary with the following key-value pairs:
                ptr: torch.Tensor of size [num_graphs + 1] with the indexes
                    of the first node of each graph in the batch.
                features: dictionary with features from different layers.
                std: float of per-atom data standard deviation.
                use_var_scale_factor: bool indicatinf wheter to scale variance 
                    with var_scale_factor.

        Returns:
            Variance per graph.
        """
        ptr = input_['ptr']
        features = input_['features']
        std = input_['std']
        use_var_scale_factor = input_['use_var_scale_factor']

        all_features = torch.cat([
            f[0] if isinstance(f, (tuple, list)) else f 
            for f in features.values()
        ], dim=-1)
        variance = self.linear_1(all_features)
        variance = F.silu(variance)
        variance = stable_softplus(self.linear_2(variance))   # [num_nodes, 1]

        variance_per_graph = segment_sum_csr(variance, ptr)   # [num_graphs, 1]

        scale = std**2
        if use_var_scale_factor:
            scale = scale*stable_softplus(self.var_scale_factor)

        return scale*variance_per_graph



class PaiNNConstantVariancePrediction(nn.Module):

    def __init__(self, sigma_obs: float = 1.0):
        super().__init__()
        self.register_buffer('sigma_obs', torch.tensor(sigma_obs))
        self.var_scale_factor = nn.Parameter(
            inverse_stable_softplus(torch.tensor(1.0)),
            requires_grad=False
        )


    def forward(self, input_: PredictionInput) -> torch.Tensor:
        var = self.sigma_obs**2
        if input_['use_var_scale_factor']:
            var = var*stable_softplus(self.var_scale_factor)

        return var


class PainNNEnsembleVariancePrediction(nn.Module):
    def __init__(self, ensemble_size: int):
        super().__init__()
        self.ensemble_size = ensemble_size

        self.variance_heads = nn.ModuleList(
            PaiNNLinearVariancePrediction() for k in range(self.ensemble_size)
        )


    def forward(self, input_: PredictionInput) -> torch.Tensor:
        features = input_['features']

        variance = list()
        for k, (feature_key, feature) in enumerate(features.items()):
            input_['features'] = {feature_key: feature}
            variance_k = self.variance_heads[k](input_)
            variance.append(variance_k)

        input_['features'] = features

        variance = torch.cat(variance, dim=1)
        return variance
