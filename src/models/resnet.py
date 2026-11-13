import torch
import torchvision
import torch.nn as nn
from copy import deepcopy
from typing import TypedDict
from src.utils import stable_softplus, inverse_stable_softplus


class ResNetPredictionInput(TypedDict):
    last_layer_features: torch.Tensor
    features: dict
    std: float
    use_var_scale_factor: bool


class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
        self.model.fc = nn.Identity()

    def forward(self, input_: torch.Tensor) -> ResNetPredictionInput:
        return {'last_layer_features': self.model(input_)}


class ResNetHead(nn.Module):
    """
    Regression head for ResNet.
    """

    def __init__(
        self, 
        mean: float = 0.,
        std: float = 1.,
    ) -> None:
        """
        Args:
            mean: torch.Tensor with mean value to shift predictions by.
            std: torch.Tensor with standard deviation to scale predictions by.
        """
        super().__init__()
        self.register_buffer(
            'mean',
            mean.clone().detach() if isinstance(mean, torch.Tensor) else torch.tensor(mean)
        )
        self.register_buffer(
            'std',
            std.clone().detach() if isinstance(std, torch.Tensor) else torch.tensor(std)
        )
        self.linear = nn.LazyLinear(1)


    def forward(self, input_: ResNetPredictionInput) -> torch.Tensor:
        """
        Forward pass of ResNet for regression.

        Args:
            input_: Dictionary with the following key-value pairs:
                last_layer_features: torch.Tensor of size [num_images, 
                    num_last_layer_features].

        Returns:
            A tensor of size [num_images, 1] with predictions for
            each image.
        """
        ll_features = input_['last_layer_features']
        predictions = self.linear(ll_features) # [num_images, 1]

        # Scale and shift
        predictions = self.std*predictions + self.mean

        return predictions


class ResNetLinearVariancePrediction(nn.Module):
    """
    Variance prediction head for ResNet that predicts variance as a linear 
    function of features.
    """

    def __init__(self) -> None:
       super().__init__()
       self.linear = nn.LazyLinear(1)

       # Scale factor that can be fitted with calibration data
       self.var_scale_factor = nn.Parameter(
            inverse_stable_softplus(torch.tensor(1.0)),
            requires_grad=False
        )


    def forward(self, input_: ResNetPredictionInput) -> torch.Tensor:
        """
        Args:
            input_: Dictionary with the following key-value pairs:
                features: dictionary with features from different layers.
                std: float of target standard deviation.
                use_var_scale_factor: bool indicating wheter to scale variance 
                    with var_scale_factor.

        Returns:
            Variance per image.
        """
        features = input_['features']
        std = input_['std']
        use_var_scale_factor = input_['use_var_scale_factor']

        all_features = list()
        for feature in features.values():
            if feature.dim() == 4: # [num_images, num_channels, height, width]
                # Average over spatial dimensions
                feature = torch.mean(feature, dim=(2, 3))

            all_features.append(feature)
        all_features = torch.cat(all_features, dim=-1)

        variance = stable_softplus(self.linear(all_features)) # [num_images, 1]

        scale = std**2
        if use_var_scale_factor:
            scale = scale*stable_softplus(self.var_scale_factor)

        return scale*variance


class ResNetConstantVariancePrediction(nn.Module):

    def __init__(self, sigma_obs: float = 1.0):
        super().__init__()
        self.register_buffer('sigma_obs', torch.tensor(sigma_obs))
        self.var_scale_factor = nn.Parameter(
            inverse_stable_softplus(torch.tensor(1.0)),
            requires_grad=False
        )


    def forward(self, input_: ResNetPredictionInput) -> torch.Tensor:
        var = self.sigma_obs**2
        if input_['use_var_scale_factor']:
            var = var*stable_softplus(self.var_scale_factor)

        return var


class ResNetEnsembleVariancePrediction(nn.Module):
    def __init__(self, ensemble_size: int):
        super().__init__()
        self.ensemble_size = ensemble_size

        self.variance_heads = nn.ModuleList(
            ResNetLinearVariancePrediction() for k in range(self.ensemble_size)
        )


    def forward(self, input_: ResNetPredictionInput) -> torch.Tensor:

        variance = list()
        features = deepcopy(input_['features'])
        for k, (feature_key, feature) in enumerate(features.items()):
            input_['features'] = {feature_key: feature}
            variance_k = self.variance_heads[k](input_)
            variance.append(variance_k)

        variance = torch.cat(variance, dim=1)
        return variance