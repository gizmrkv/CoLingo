import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torchtyping import TensorType


class TransformerEncoder(nn.Module):
    BATCH = "batch"
    LENGTH = "length"
    OUTPUT_DIM = "output_dim"

    def __init__(
        self,
        n_values: int,
        output_dim: int,
        embed_dim: int,
        n_heads: int,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        n_layers: int = 6,
    ) -> None:
        super().__init__()
        self._n_values = n_values
        self._output_dim = output_dim
        self._embed_dim = embed_dim
        self._n_heads = n_heads
        self._ff_dim = ff_dim
        self._dropout = dropout
        self._activation = activation
        self._layer_norm_eps = layer_norm_eps
        self._batch_first = batch_first
        self._norm_first = norm_first
        self._n_layers = n_layers

        self._embed = nn.Embedding(n_values, embed_dim)
        self._transformer = Transformer(
            dim=embed_dim,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            n_layers=n_layers,
        )
        self._linear = nn.Linear(embed_dim, output_dim)

    def forward(
        self, x: TensorType[BATCH, LENGTH]
    ) -> TensorType[BATCH, LENGTH, OUTPUT_DIM]:
        x = self._embed(x)
        x = self._transformer(x)
        x = x.sum(dim=1)
        return self._linear(x)


class TransformerDecoder(nn.Module):
    BATCH = "batch"
    INPUT_DIM = "input_dim"
    LENGTH = "length"
    N_VALUES = "n_values"

    def __init__(
        self,
        input_dim: int,
        length: int,
        n_values: int,
        embed_dim: int,
        n_heads: int,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        n_layers: int = 6,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._length = length
        self._n_values = n_values
        self._embed_dim = embed_dim
        self._n_heads = n_heads
        self._ff_dim = ff_dim
        self._dropout = dropout
        self._activation = activation
        self._layer_norm_eps = layer_norm_eps
        self._batch_first = batch_first
        self._norm_first = norm_first
        self._n_layers = n_layers

        self._pre_linear = nn.Linear(input_dim, embed_dim)
        self._transformer = Transformer(
            dim=embed_dim,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            n_layers=n_layers,
        )
        self._post_linear = nn.Linear(embed_dim, n_values)

    def forward(
        self, x: TensorType[BATCH, INPUT_DIM, float]
    ) -> tuple[
        TensorType[BATCH, LENGTH, int], TensorType[BATCH, LENGTH, N_VALUES, float]
    ]:
        x = self._pre_linear(x)
        x = x.unsqueeze(dim=1).repeat_interleave(self._length, dim=1)
        x = self._transformer(x)
        logits = self._post_linear(x)

        if self.training:
            distr = Categorical(logits=logits)
            x = distr.sample()
            return x, logits
        else:
            return logits.argmax(dim=-1), logits


class Transformer(nn.Module):
    BATCH = "batch"
    LENGTH = "length"
    DIM = "dim"

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        n_layers: int = 6,
    ) -> None:
        super().__init__()
        self._pos_encoder = PositionalEncoding(dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self._model = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self, x: TensorType[BATCH, LENGTH, DIM]
    ) -> TensorType[BATCH, LENGTH, DIM]:
        x = self._pos_encoder(x)
        return self._model(x)


class PositionalEncoding(nn.Module):
    BATCH = "batch"
    LENGTH = "length"
    DIM = "dim"

    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self._dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, 1, dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(
        self, x: TensorType[BATCH, LENGTH, DIM]
    ) -> TensorType[BATCH, LENGTH, DIM]:
        x = x + self.pe[: x.size(0)]  # type: ignore
        return self._dropout(x)
