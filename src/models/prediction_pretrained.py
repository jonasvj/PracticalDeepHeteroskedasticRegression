import torch
import torch.nn as nn
from typing import List
from torch_scatter import segment_sum_csr
from src.models.painn.prediction import PredictionInput
from src.utils import stable_softplus, inverse_stable_softplus


class FeatureModelLinearVariancePrediction(nn.Module):

    def __init__(self, feature_keys: List[str]) -> None:
       super().__init__()
       self.feature_keys = feature_keys
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
        Returns:
            Variance per graph.
        """
        ptr = input_['ptr']
        features = input_['features']
        use_var_scale_factor = input_['use_var_scale_factor']

        all_features = torch.cat(
            [features[key] for key in self.feature_keys],
            dim=-1
        )
        variance = stable_softplus(self.linear(all_features)) # [num_nodes, 1]
        variance_per_graph = segment_sum_csr(variance, ptr)   # [num_graphs, 1]

        if use_var_scale_factor:
            variance_per_graph = variance_per_graph*stable_softplus(self.var_scale_factor)

        return variance_per_graph


class FeatureModelConstantVariancePrediction(nn.Module):

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


class FeatureModelEnsembleVariancePrediction(nn.Module):
    def __init__(self, feature_keys: List[str]):
        super().__init__()
        self.feature_keys = feature_keys

        self.variance_heads = nn.ModuleList(
            FeatureModelLinearVariancePrediction(feature_keys=[key]) 
            for key in self.feature_keys
        )


    def forward(self, input_: PredictionInput) -> torch.Tensor:
        variance = list()
        for var_head in self.variance_heads:
            variance_k = var_head(input_)
            variance.append(variance_k)

        variance = torch.cat(variance, dim=1)
        return variance
