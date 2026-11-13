import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Optional, Any, Tuple, Literal
from torch_scatter import segment_sum_csr
from src.models.painn.representation import BackboneInput


class PaiNN(nn.Module):
    """
    Module that wraps together backbone, prediction head, feature extractor, 
    and variance head for PaiNN models.
    """
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module, 
        feature_extractor: nn.Module,
        variance_head: nn.Module,
        name: str = 'painn',
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.variance_head = variance_head
        self.name = name

        self.std = head.std if hasattr(head, 'std') else 1.0

        self.feature_extractor.wrap_model(
            nn.Sequential(OrderedDict([('backbone', backbone), ('head', head)]))
        )


    def forward(
        self,
        input_: BackboneInput,
        use_var_scale_factor: bool = True,
        mean: Optional[torch.Tensor] = None,
        features: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mean is None or features is None:
            mean, features = self.feature_extractor(input_)
        mean = mean.squeeze(-1)

        input_.update({
            'features': features,
            'std': self.std,
            'use_var_scale_factor': use_var_scale_factor
        })

        var = self.variance_head(input_)
        var = torch.ones_like(mean)*var if var.dim() == 0 else var.squeeze(-1)

        return mean, var


    def normalize_targets(self, y: torch.Tensor, z: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
        """
        Transform physical-space targets (or predictions) into the normalized
        space used by the network, i.e. undo the denormalization applied inside
        PaiNNAtomwisePrediction.forward:

            y_norm = (y_phys - n_atoms * mu - sum_i atom_refs[z_i]) / sigma

        Args:
            y:   Tensor of shape [num_graphs] in physical (raw target) space.
            z:   Tensor of shape [num_nodes] with atom types.
            ptr: Tensor of shape [num_graphs + 1] with graph boundaries.

        Returns:
            Tensor of shape [num_graphs] in normalized network-output space.
        """
        head = self.feature_extractor.model.head
        n_atoms = (ptr[1:] - ptr[:-1]).float()
        offset = head.mean * n_atoms
        if head.atom_refs is not None:
            offset = offset + segment_sum_csr(head.atom_refs(z), ptr).squeeze(-1)
        return (y - offset) / head.std


class FeatureModel(nn.Module):
    """
    Wrapper for UMA and AllScAIP models with precomputed features.
    """
    def __init__(
        self,
        variance_head: nn.Module,
        name: Literal['uma', 'allscaip'] = 'uma',
    ):
        super().__init__()
        self.variance_head = variance_head
        self.name = name


    def forward(
        self,
        input_: BackboneInput,
        use_var_scale_factor: bool = True,
        mean: Optional[torch.Tensor] = None,
        features: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = input_['mean'].squeeze(-1)
        input_.update({'use_var_scale_factor': use_var_scale_factor})

        var = self.variance_head(input_)
        var = torch.ones_like(mean)*var if var.dim() == 0 else var.squeeze(-1)

        return mean, var


class ResNet(nn.Module):
    """
    Module that wraps together backbone, prediction head, feature extractor, 
    and variance head for ResNet models.
    """
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module, 
        feature_extractor: nn.Module,
        variance_head: nn.Module,
        name: str = 'resnet',
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.variance_head = variance_head
        self.name = name

        self.std = head.std if hasattr(head, 'std') else 1.0

        self.feature_extractor.wrap_model(
            nn.Sequential(OrderedDict([('backbone', backbone), ('head', head)]))
        )


    def forward(
        self,
        input_: torch.Tensor,
        use_var_scale_factor: bool = True,
        mean: Optional[torch.Tensor] = None,
        features: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mean is None or features is None:
            mean, features = self.feature_extractor(input_)
        mean = mean.squeeze(-1)

        input_ ={
            'features': features,
            'std': self.std,
            'use_var_scale_factor': use_var_scale_factor
        }

        var = self.variance_head(input_)
        var = torch.ones_like(mean)*var if var.dim() == 0 else var.squeeze(-1)

        return mean, var


class FTTransformer(nn.Module):
    """
    Module that wraps together backbone, prediction head, feature extractor, 
    and variance head for FT-Transformer models.
    """
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module, 
        feature_extractor: nn.Module,
        variance_head: nn.Module,
        name: str = 'ft_transformer',
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.variance_head = variance_head
        self.name = name

        self.std = head.std if hasattr(head, 'std') else 1.0

        self.feature_extractor.wrap_model(
            nn.Sequential(OrderedDict([('backbone', backbone), ('head', head)]))
        )


    def forward(
        self,
        input_: torch.Tensor,
        use_var_scale_factor: bool = True,
        mean: Optional[torch.Tensor] = None,
        features: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mean is None or features is None:
            mean, features = self.feature_extractor(input_)
        mean = mean.squeeze(-1)

        input_ = {
            'features': features,
            'std': self.std,
            'use_var_scale_factor': use_var_scale_factor
        }

        var = self.variance_head(input_)
        var = torch.ones_like(mean)*var if var.dim() == 0 else var.squeeze(-1)

        return mean, var
