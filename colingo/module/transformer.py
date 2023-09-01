import math

import torch
from torch import nn
from torch.distributions import Categorical
from torchtyping import TensorType


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


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        embed_dim: int,
        n_heads: int = 1,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        n_layers: int = 6,
        is_causal: bool = False,
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
        self.is_causal = is_causal

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.linear = nn.Linear(embed_dim, output_dim)
        self.sos_embed = nn.Parameter(torch.randn(embed_dim))

    def forward(
        self, input: TensorType[..., int]
    ) -> TensorType[..., "output_dim", float]:
        embed = self.embed(input)
        embed = torch.cat([self.sos_embed.expand(embed.shape[0], 1, -1), embed], dim=1)
        embed = embed * math.sqrt(self.embed_dim)
        embed = self.pos_encoder(embed)
        output = self.encoder(embed, is_causal=self.is_causal)
        # As the input to the agent, we take the embedding for the first symbol
        # which is always the special <sos> one.
        output = self.linear(output[:, 0, :])
        return output


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        max_len: int,
        vocab_size: int,
        embed_dim: int,
        n_heads: int = 1,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        n_layers: int = 6,
    ):
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

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.proj1 = nn.Linear(input_dim, embed_dim)
        self.proj2 = nn.Linear(embed_dim, vocab_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.sos_embed = nn.Parameter(torch.randn(embed_dim))

    def decode_standard(
        self,
        latent: TensorType[..., "input_dim", float],
        input: TensorType[..., int] | None = None,
        message: TensorType[..., "max_len", int] | None = None,
    ) -> TensorType[..., "max_len", "vocab_size", float]:
        memory = self.proj1(latent)
        memory = memory.unsqueeze(1)
        input = self.sos_embed.expand(latent.shape[0], 1, -1)
        symbols_list = []
        logits_list = []
        for i in range(self.max_len):
            logits_step = self.decoder(self.pos_encoder(input), memory)
            logits_step = self.proj2(logits_step[:, -1, :])

            if self.training:
                distr = Categorical(logits=logits_step)
                output = distr.sample()
            else:
                output = logits_step.argmax(dim=-1)

            embed = math.sqrt(self.embed_dim) * self.embed(
                output if message is None else message[:, 0]
            )
            input = torch.cat([input, embed.unsqueeze(1)], dim=1)

            symbols_list.append(output)
            logits_list.append(logits_step)

        symbols = torch.stack(symbols_list, dim=1)
        logits = torch.stack(logits_list, dim=1)
        return symbols, logits

    def forward(
        self,
        latent: TensorType[..., "input_dim", float],
        input: TensorType[..., int] | None = None,
        message: TensorType[..., "max_len", int] | None = None,
    ) -> TensorType[..., "max_len", "vocab_size", float]:
        return self.decode_standard(latent, input, message)
