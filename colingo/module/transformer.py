import math

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchtyping import TensorType


class TransformerEncoder(nn.Module):
    """
    Implementation of a Transformer encoder.

    Args:
        input_dim (int): Dimension of the input data.
        n_heads (int): Number of attention heads.
        ff_dim (int, optional): Dimension of the feed-forward network. Defaults to 2048.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        activation (str, optional): Activation function for feed-forward network. Defaults to "relu".
        layer_norm_eps (float, optional): Epsilon value for layer normalization. Defaults to 1e-5.
        norm_first (bool, optional): Apply layer normalization before attention in encoder layers. Defaults to False.
        n_layers (int, optional): Number of encoder layers. Defaults to 6.
        max_len (int, optional): Maximum positional encoding length. Defaults to 5000.
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        n_layers: int = 6,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.norm_first = norm_first
        self.n_layers = n_layers

        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        activations = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
        }
        layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activations[activation](),
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self, x: TensorType[..., "input_dim", float]
    ) -> TensorType[..., "input_dim", float]:
        x = self.pos_encoder(x)
        return self.encoder(x)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.

    Args:
        input_dim (int): Dimension of the input data.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        max_len (int, optional): Maximum positional encoding length. Defaults to 5000.
    """

    def __init__(self, input_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, input_dim, 2) * (-math.log(10000.0) / input_dim)
        )
        pe = torch.zeros(1, max_len, input_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(
        self, x: TensorType[..., "input_dim", float]
    ) -> TensorType[..., "input_dim", float]:
        x = x + self.pe[:, : x.size(1), :]  # type: ignore
        return self.dropout(x)


class IntSequenceTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        embed_dim: int,
        n_heads: int,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        n_layers: int = 6,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.norm_first = norm_first
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = TransformerEncoder(
            input_dim=embed_dim,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            n_layers=n_layers,
        )
        self.linear = nn.Linear(embed_dim, output_dim)

    def forward(
        self, input: TensorType[..., int]
    ) -> TensorType[..., "embed_dim", float]:
        embed = self.embed(input)
        output = self.encoder(embed)
        output = self.linear(output.sum(dim=1))
        return output


class IntSequenceTransformerDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        max_len: int,
        vocab_size: int,
        embed_dim: int,
        n_heads: int,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        n_layers: int = 6,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.norm_first = norm_first
        self.n_layers = n_layers

        self.pre_linear = nn.Linear(input_dim, max_len * embed_dim)
        self.pro_linear = nn.Linear(embed_dim, vocab_size)
        self.encoder = TransformerEncoder(
            input_dim=embed_dim,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            n_layers=n_layers,
        )

    def forward(
        self, latent: TensorType[..., "input_dim", float]
    ) -> TensorType[..., "max_len", "vocab_size", float]:
        embed = self.pre_linear(latent)
        embed = embed.view(-1, self.max_len, self.embed_dim)
        output = self.encoder(embed)
        logits = self.pro_linear(output)
        if self.training:
            distr = Categorical(logits=logits)
            return distr.sample(), logits
        else:
            return logits.argmax(dim=-1), logits
