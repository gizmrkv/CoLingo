import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torchtyping import TensorType


class DiscSeqRNNEncoder(nn.Module):
    def __init__(
        self,
        length: int,
        n_values: int,
        output_dim: int,
        hidden_dim: int,
        embed_dim: int,
        rnn_type: str = "rnn",
        n_layers: int = 1,
    ):
        super().__init__()
        self.length = length
        self.n_values = n_values
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.rnn_type = rnn_type
        self.n_layers = n_layers

        self.embed = nn.Embedding(n_values, embed_dim)
        self.hidden2output = nn.Linear(hidden_dim, output_dim)

        rnn_type = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.rnn = rnn_type(embed_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, x: TensorType["batch", "length", int]):
        x = self.embed(x)
        _, h = self.rnn(x)
        if isinstance(self.rnn, nn.LSTM):
            h, _ = h
        h = h[-1]
        h = self.hidden2output(h)
        return h


class DiscSeqRNNDecoder(nn.Module):
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

        rnn_type = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.rnn = rnn_type(embed_dim, hidden_dim, n_layers, batch_first=True)

        self.sos_embed = nn.Parameter(torch.randn(embed_dim))

    def forward(
        self,
        x: TensorType["batch", "input_dim", float],
        input: TensorType["batch", "length", int] | None = None,
    ):
        h = torch.stack([i2h(x) for i2h in self.input2hidden], dim=0)
        if isinstance(self.rnn, nn.LSTM):
            h = (h, h)

        output = []
        logits = []

        if input is None:
            i = self.sos_embed.repeat(x.shape[0], 1, 1)
            for _ in range(self.length):
                # i = i.unsqueeze(1)
                y, h = self.rnn(i, h)
                logit = self.hidden2output(y)

                if self.training:
                    distr = Categorical(logits=logit)
                    x = distr.sample()
                else:
                    x = logit.argmax(dim=-1)

                i = self.embed(x)
                output.append(x)
                if self.training:
                    logits.append(logit)

            output = torch.cat(output, dim=1)
            logits = torch.cat(logits, dim=1)

        else:
            i = self.embed(input)
            y, _ = self.rnn(i, h)
            logits = self.hidden2output(y)

            if self.training:
                distr = Categorical(logits=logits)
                output = distr.sample()
            else:
                output = logits.argmax(dim=-1)

        if self.training:
            return output, logits
        else:
            return output
