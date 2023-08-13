from typing import List, Literal

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torchtyping import TensorType


class RNNEncoder(nn.Module):
    BATCH = "batch"
    LENGTH = "length"
    OUTPUT_DIM = "output_dim"

    def __init__(
        self,
        n_values: int,
        output_dim: int,
        embed_dim: int,
        hidden_dim: int,
        rnn_type: str = "rnn",
        n_layers: int = 1,
    ):
        super().__init__()
        self._n_values = n_values
        self._output_dim = output_dim
        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._rnn_type = rnn_type
        self._n_layers = n_layers

        self._embed = nn.Embedding(n_values, embed_dim)

        rnns = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        self._rnn = rnns[rnn_type.lower()](
            embed_dim, hidden_dim, n_layers, batch_first=True
        )

        self._linear = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: TensorType[BATCH, LENGTH, int]
    ) -> TensorType[BATCH, OUTPUT_DIM, float]:
        x = self._embed(x)
        _, h = self._rnn(x)
        if isinstance(self._rnn, nn.LSTM):
            h, _ = h
        h = h[-1]
        h = self._linear(h)
        return h


class RNNDecoder(nn.Module):
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
        hidden_dim: int,
        rnn_type: str = "rnn",
        n_layers: int = 1,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._length = length
        self._n_values = n_values
        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._rnn_type = rnn_type
        self._n_layers = n_layers

        self._embed = nn.Embedding(n_values, embed_dim)
        self._pre_linear = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim) for _ in range(n_layers)]
        )
        self._pro_linear = nn.Linear(hidden_dim, n_values)

        rnns = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        self._rnn = rnns[rnn_type.lower()](
            embed_dim, hidden_dim, n_layers, batch_first=True
        )

        self.sos_embed = nn.Parameter(torch.randn(embed_dim))

    def forward(
        self,
        x: TensorType[BATCH, INPUT_DIM, float],
        input: TensorType[BATCH, LENGTH, int] | None = None,
    ) -> tuple[
        TensorType[BATCH, LENGTH, int], TensorType[BATCH, LENGTH, N_VALUES, float]
    ]:
        h = torch.stack([linear(x) for linear in self._pre_linear])
        if isinstance(self._rnn, nn.LSTM):
            h = (h, torch.zeros_like(h))  # type: ignore

        if input is None:
            outputs: List[torch.Tensor] = []
            logitss = []
            i = self.sos_embed.repeat(x.shape[0], 1, 1)
            for _ in range(self._length):
                y, h = self._rnn(i, h)
                logit = self._pro_linear(y)

                if self.training:
                    distr = Categorical(logits=logit)
                    x = distr.sample()
                else:
                    x = logit.argmax(dim=-1)

                i = self._embed(x)
                outputs.append(x)
                logitss.append(logit)

            output = torch.cat(outputs, dim=1)
            logits = torch.cat(logitss, dim=1)

        else:
            i = self._embed(input)
            y, _ = self._rnn(i, h)
            logits = self._pro_linear(y)

            if self.training:
                distr = Categorical(logits=logits)
                output = distr.sample()
            else:
                output = logits.argmax(dim=-1)

        return output, logits
