import torch
from torch import nn
from torch.distributions import Categorical
from torchtyping import TensorType


class IntSequenceRNNEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        embed_dim: int,
        hidden_dim: int,
        rnn_type: str = "rnn",
        n_layers: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()](
            embed_dim, hidden_dim, n_layers, batch_first=True
        )

        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, input: TensorType[..., int]
    ) -> TensorType[..., "output_dim", float]:
        embed = self.embed(input)
        _, hidden = self.rnn(embed)
        if isinstance(self.rnn, nn.LSTM):
            hidden, _ = hidden
        hidden = hidden[-1]
        # The output is the dimension of the hidden state of the last layer
        output = self.linear(hidden)
        return output


class IntSequenceRNNDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        max_len: int,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        rnn_type: str = "rnn",
        n_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pre_linear = nn.Linear(input_dim, hidden_dim)
        self.pro_linear = nn.Linear(hidden_dim, vocab_size)

        self.rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()](
            embed_dim, hidden_dim, n_layers, batch_first=True
        )

        self.sos_embed = nn.Parameter(torch.randn(embed_dim))

    def forward(
        self,
        latent: TensorType[..., "input_dim", float],
    ) -> TensorType[..., "max_len", "vocab_size", float]:
        # Adjust the dimension of the input, put it in hidden, and duplicate it
        hidden = self.pre_linear(latent)
        hidden = hidden.repeat(self.n_layers, 1, 1)
        if isinstance(self.rnn, nn.LSTM):
            hidden = (hidden, torch.zeros_like(hidden))

        logits_seq = []
        input = self.sos_embed.repeat(latent.shape[0], 1, 1)
        for _ in range(self.max_len):
            logits, hidden = self.rnn(input, hidden)
            logits = self.pro_linear(logits)

            if self.training:
                distr = Categorical(logits=logits)
                output = distr.sample()
            else:
                output = logits.argmax(dim=-1)

            input = self.embed(output)
            logits_seq.append(logits)

        logits = torch.cat(logits_seq, dim=1)
        if self.training:
            distr = Categorical(logits=logits)
            return distr.sample(), logits
        else:
            return logits.argmax(dim=-1), logits
