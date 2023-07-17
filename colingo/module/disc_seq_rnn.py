from typing import Literal

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torchtyping import TensorType


class DiscSeqRNNEncoder(nn.Module):
    BATCH = "batch"
    LENGTH = "length"
    OUTPUT_DIM = "output_dim"

    def __init__(
        self,
        n_values: int,
        output_dim: int,
        hidden_dim: int,
        embed_dim: int,
        rnn_type: Literal["rnn", "lstm", "gru"] = "rnn",
        n_layers: int = 1,
    ):
        super().__init__()
        self.n_values = n_values
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.rnn_type = rnn_type
        self.n_layers = n_layers

        self.embed = nn.Embedding(n_values, embed_dim)
        self.hidden2output = nn.Linear(hidden_dim, output_dim)

        rnns = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        self.rnn = rnns[rnn_type.lower()](
            embed_dim, hidden_dim, n_layers, batch_first=True
        )

    def forward(
        self, x: TensorType[BATCH, LENGTH, int]
    ) -> TensorType[BATCH, OUTPUT_DIM, float]:
        x = self.embed(x)
        _, h = self.rnn(x)
        if isinstance(self.rnn, nn.LSTM):
            h, _ = h
        h = h[-1]
        h = self.hidden2output(h)
        return h


class DiscSeqRNNDecoder(nn.Module):
    BATCH = "batch"
    INPUT_DIM = "input_dim"
    LENGTH = "length"
    N_VALUES = "n_values"

    def __init__(
        self,
        length: int,
        n_values: int,
        input_dim: int,
        hidden_dim: int,
        embed_dim: int,
        rnn_type: str = "rnn",
        n_layers: int = 1,
    ):
        super().__init__()
        self.length = length
        self.n_values = n_values
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.rnn_type = rnn_type
        self.n_layers = n_layers

        self.embed = nn.Embedding(n_values, embed_dim)
        self.input2hidden = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.hidden2output = nn.Linear(hidden_dim, n_values)

        rnns = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        self.rnn = rnns[rnn_type.lower()](
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
        h = torch.stack([i2h(x) for i2h in self.input2hidden])
        if isinstance(self.rnn, nn.LSTM):
            c = (h, torch.zeros_like(h))

        if input is None:
            outputs: list[torch.Tensor] = []
            logitss = []
            i = self.sos_embed.repeat(x.shape[0], 1, 1)
            for _ in range(self.length):
                if isinstance(self.rnn, nn.LSTM):
                    y, (h, c) = self.rnn(i, (h, c))
                else:
                    y, h = self.rnn(i, h)
                logit = self.hidden2output(y)

                if self.training:
                    distr = Categorical(logits=logit)
                    x = distr.sample()
                else:
                    x = logit.argmax(dim=-1)

                i = self.embed(x)
                outputs.append(x)
                logitss.append(logit)

            output = torch.cat(outputs, dim=1)
            logits = torch.cat(logitss, dim=1)

        else:
            i = self.embed(input)
            y, _ = self.rnn(i, h)
            logits = self.hidden2output(y)

            if self.training:
                distr = Categorical(logits=logits)
                output = distr.sample()
            else:
                output = logits.argmax(dim=-1)

        return output, logits
