import math
import typing as ty
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor
from typing import TypedDict
from src.utils import stable_softplus, inverse_stable_softplus


class FTTransformerPredictionInput(TypedDict):
    last_layer_features: torch.Tensor
    features: dict
    std: float
    use_var_scale_factor: bool


class NumericalTokenizer(nn.Module):
    """Tokenizer for numerical features using element-wise multiplication.

    Each numerical feature is mapped to a d_token-dimensional embedding via
    a learned weight vector (element-wise multiplication).  A [CLS] token is
    prepended automatically.
    """

    def __init__(self, d_numerical: int, d_token: int, bias: bool = True) -> None:
        super().__init__()
        # +1 for [CLS] token
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_numerical, d_token)) if bias else None
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight)

    def forward(self, x_num: Tensor) -> Tensor:
        # Prepend a column of ones for the [CLS] token
        x = torch.cat(
            [torch.ones(len(x_num), 1, device=x_num.device), x_num],
            dim=1,
        )
        x = self.weight[None] * x[:, :, None]  # [B, 1+d_num, d_token]
        if self.bias is not None:
            bias = torch.cat(
                [torch.zeros(1, self.bias.shape[1], device=x.device), self.bias]
            )
            x = x + bias[None]
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']
        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


def _get_activation_fn(name: str):
    return (
        F.relu if name == 'relu'
        else F.gelu if name == 'gelu'
        else (lambda x: F.gelu(x, approximate='tanh')) if name == 'geglu'
        else (lambda x: x * F.sigmoid(x)) if name == 'swish'
        else _reglu if name == 'reglu'
        else None
    )


def _reglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def _get_nonglu_activation_fn(name: str):
    return (
        F.relu if name in ('relu', 'reglu')
        else F.gelu if name in ('gelu', 'geglu')
        else (lambda x: x * F.sigmoid(x)) if name == 'swish'
        else None
    )


class TransformerBlock(nn.Module):
    """A single Transformer block (attention + FFN) with residual connections.

    Wrapped as its own ``nn.Module`` so that PyTorch forward hooks (used by
    ``FeatureExtractor``) can be attached per-block.
    """

    def __init__(
        self,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        is_first_layer: bool,
    ) -> None:
        super().__init__()
        self.prenormalization = prenormalization
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout

        d_hidden = int(d_token * d_ffn_factor)

        self.attention = MultiheadAttention(
            d_token, n_heads, attention_dropout, initialization,
        )
        self.linear0 = nn.Linear(
            d_token, d_hidden * (2 if activation.endswith('glu') else 1)
        )
        self.linear1 = nn.Linear(d_hidden, d_token)
        self.norm1 = nn.LayerNorm(d_token)

        if not prenormalization or not is_first_layer:
            self.norm0 = nn.LayerNorm(d_token)
        else:
            self.norm0 = None

        self.activation = _get_activation_fn(activation)

    def _start_residual(self, x: Tensor, norm: ty.Optional[nn.LayerNorm]) -> Tensor:
        x_residual = x
        if self.prenormalization and norm is not None:
            x_residual = norm(x_residual)
        return x_residual

    def _end_residual(
        self, x: Tensor, x_residual: Tensor, norm: ty.Optional[nn.LayerNorm]
    ) -> Tensor:
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization and norm is not None:
            x = norm(x)
        return x

    def forward(self, x: Tensor, is_last_layer: bool = False) -> Tensor:
        # --- Attention sublayer ---
        x_residual = self._start_residual(x, self.norm0)
        x_residual = self.attention(
            x_residual[:, :1] if is_last_layer else x_residual,
            x_residual,
        )
        if is_last_layer:
            x = x[:, :x_residual.shape[1]]
        x = self._end_residual(x, x_residual, self.norm0 if not self.prenormalization else None)

        # --- FFN sublayer ---
        x_residual = self._start_residual(x, self.norm1)
        x_residual = self.linear0(x_residual)
        x_residual = self.activation(x_residual)
        if self.ffn_dropout:
            x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
        x_residual = self.linear1(x_residual)
        x = self._end_residual(x, x_residual, self.norm1)

        return x


class FTTransformerBackbone(nn.Module):
    """FT-Transformer backbone: NumericalTokenizer + stack of TransformerBlocks.

    Returns a dict ``{'last_layer_features': Tensor}`` so that an
    ``FTTransformerHead`` can be composed via the same ``FeatureExtractor``
    pipeline used for ``ResNet``.
    """

    def __init__(
        self,
        d_numerical: int,
        n_layers: int = 3,
        d_token: int = 192,
        n_heads: int = 8,
        d_ffn_factor: float = 1.333333333333333,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        activation: str = 'reglu',
        prenormalization: bool = True,
        initialization: str = 'kaiming',
        token_bias: bool = True,
    ) -> None:
        super().__init__()
        self.prenormalization = prenormalization

        self.tokenizer = NumericalTokenizer(d_numerical, d_token, token_bias)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_token=d_token,
                n_heads=n_heads,
                d_ffn_factor=d_ffn_factor,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout,
                activation=activation,
                prenormalization=prenormalization,
                initialization=initialization,
                is_first_layer=(i == 0),
            )
            for i in range(n_layers)
        ])

        self.last_normalization = (
            nn.LayerNorm(d_token) if prenormalization else None
        )
        self.last_activation = _get_nonglu_activation_fn(activation)

    def forward(self, x_num: Tensor) -> FTTransformerPredictionInput:
        x = self.tokenizer(x_num)

        for i, block in enumerate(self.blocks):
            is_last = (i + 1 == len(self.blocks))
            x = block(x, is_last_layer=is_last)

        assert x.shape[1] == 1
        x = x[:, 0]  # [CLS] token representation
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)

        return {'last_layer_features': x}


class FTTransformerHead(nn.Module):
    """Regression head for FT-Transformer: Linear(d_token, 1) with target scaling."""

    def __init__(
        self,
        d_token: int = 192,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer(
            'mean',
            mean.clone().detach() if isinstance(mean, torch.Tensor) else torch.tensor(mean),
        )
        self.register_buffer(
            'std',
            std.clone().detach() if isinstance(std, torch.Tensor) else torch.tensor(std),
        )
        self.linear = nn.Linear(d_token, 1)

    def forward(self, input_: FTTransformerPredictionInput) -> Tensor:
        ll_features = input_['last_layer_features']
        predictions = self.linear(ll_features)  # [B, 1]
        predictions = self.std * predictions + self.mean
        return predictions


class FTTransformerLinearVariancePrediction(nn.Module):
    """Variance prediction as a linear function of intermediate features."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.LazyLinear(1)
        self.var_scale_factor = nn.Parameter(
            inverse_stable_softplus(torch.tensor(1.0)),
            requires_grad=False,
        )


    def forward(self, input_: FTTransformerPredictionInput) -> Tensor:
        features = input_['features']
        std = input_['std']
        use_var_scale_factor = input_['use_var_scale_factor']

        all_features = []
        for feature_key, feature in features.items():
            if feature.dim() == 3:  # [B, n_tokens, d_token] from transformer blocks
                feature = feature[:, 0]  # Take [CLS] token
            all_features.append(feature.flatten(start_dim=1))
        all_features = torch.cat(all_features, dim=-1)

        variance = stable_softplus(self.linear(all_features))  # [B, 1]

        scale = std ** 2
        if use_var_scale_factor:
            scale = scale * stable_softplus(self.var_scale_factor)

        return scale * variance


class FTTransformerEnsembleVariancePrediction(nn.Module):
    """Ensemble of linear variance heads, one per extracted feature representation."""

    def __init__(self, ensemble_size: int):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.variance_heads = nn.ModuleList(
            FTTransformerLinearVariancePrediction() for _ in range(self.ensemble_size)
        )

    def forward(self, input_: FTTransformerPredictionInput) -> Tensor:

        variance = []
        features = deepcopy(input_['features'])
        for k, (feature_key, feature) in enumerate(features.items()):
            input_['features'] = {feature_key: feature}
            variance_k = self.variance_heads[k](input_)
            variance.append(variance_k)

        variance = torch.cat(variance, dim=1)
        return variance


class FTTransformerConstantVariancePrediction(nn.Module):
    """Constant (homoscedastic) variance prediction for FT-Transformer."""

    def __init__(self, sigma_obs: float = 1.0):
        super().__init__()
        self.register_buffer('sigma_obs', torch.tensor(sigma_obs))
        self.var_scale_factor = nn.Parameter(
            inverse_stable_softplus(torch.tensor(1.0)),
            requires_grad=False,
        )

    def forward(self, input_: FTTransformerPredictionInput) -> Tensor:
        var = self.sigma_obs ** 2
        if input_['use_var_scale_factor']:
            var = var * stable_softplus(self.var_scale_factor)
        return var
