import torch
import torch.nn as nn
from typing import List

class FeatureExtractor(nn.Module):
    """
    Captures intermediate representations from model.
    """
    def __init__(
        self,
        module_names_to_extract: List[str] = [],
        module_indices_to_extract: List[int] = [],
        capture_module_input: List[bool] = [],
        detach_features: bool = True,
    ):
        """
        Args:
            module_names_to_extract: List of module names to capture features 
                from.
            module_indices_to_extract: List of indices to filter above list.
            capture_module_input: List of bools that indicate whether module 
                input or output should be captured. Must have same length as 
                module_names_to_extract (after applying indices) or empty. If
                empty all module outputs are captured.
            detach_features: Whether the captured features should be detached
                from computational graph.
        """
        super().__init__()
        self.module_names_to_extract = module_names_to_extract
        self.module_indices_to_extract = module_indices_to_extract
        self.capture_module_input = capture_module_input
        self.detach_features = detach_features

        # Subset list of module names with indices
        if len(self.module_indices_to_extract) > 0:
            tmp_list = [
                self.module_names_to_extract[idx] for idx in self.module_indices_to_extract
            ]
            self.module_names_to_extract = tmp_list

        if len(self.capture_module_input) > 0:
            assert len(self.capture_module_input) == len(self.module_names_to_extract)
        else:
            self.capture_module_input = [False for _ in range(len(self.module_names_to_extract))]

        self.features = {}
        self.hooks = []
        self.model = None


    def wrap_model(self, model: nn.Module):
        self.model = model

        for i, name in enumerate(self.module_names_to_extract):
            module = self.model.get_submodule(name)
            self.hooks.append(
                module.register_forward_hook(
                    self.get_features(name, capture_module_input=self.capture_module_input[i])
                )
            )


    def _detach_features(self, features):
        if isinstance(features, torch.Tensor):
            return features.detach()
        elif isinstance(features, dict):
            return {k: self._detach_features(v) for k, v in features.items()}
        elif isinstance(features, (list, tuple)):
            processed_features = [self._detach_features(f) for f in features]
            return tuple(processed_features) if isinstance(features, tuple) else processed_features
        else:
            return features


    def get_features(self, name: str, capture_module_input: bool = False):
        if capture_module_input:
            def hook(module, args, output):
                features = args[0] if len(args) == 1 else args
                self.features[name] = (
                    self._detach_features(features) if self.detach_features else features
                )
        else:
            def hook(module, args, output):
                self.features[name] = (
                    self._detach_features(output) if self.detach_features else output
                )

        return hook


    def forward(self, *args, **kwargs):
        if self.model is None:
            raise ValueError("Model not wrapped. Call 'wrap_model' with a model before using the feature extractor.")
        self.features = {}
        output = self.model(*args, **kwargs)
        return output, self.features


    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class PredictUnitFeatureExtractor(FeatureExtractor):
    """
    Class for extracting features from Fairchem's predict units
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_unit = None


    def wrap_model(self, predict_unit):
        self.predict_unit = predict_unit
        return super().wrap_model(self.predict_unit.model)


    def predict(self, *args, **kwargs):
        if self.predict_unit is None:
            raise ValueError("PredictUnit not wrapped. Call 'wrap_model' with a predict unit before using the feature extractor.")
        self.features = {}
        output = self.predict_unit.predict(*args, **kwargs)
        return output, self.features
