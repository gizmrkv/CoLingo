import torch as th
from torch.distributions import Categorical
from collections import namedtuple

from . import util


class SingleWordModel(th.nn.Module):
    Auxiliary = namedtuple("Auxiliary", ["log_prob", "entropy"])

    def __init__(
        self, n_attributes: int, n_values: int, vocab_size: int, hidden_size: int = 64
    ):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.fc1 = th.nn.Sequential(
            th.nn.Linear(n_attributes * n_values, hidden_size),
            th.nn.LeakyReLU(),
            th.nn.Linear(hidden_size, hidden_size),
            th.nn.LeakyReLU(),
            th.nn.Linear(hidden_size, vocab_size),
        )
        self.fc2 = th.nn.Sequential(
            th.nn.Linear(vocab_size, hidden_size),
            th.nn.LeakyReLU(),
            th.nn.Linear(hidden_size, hidden_size),
            th.nn.LeakyReLU(),
            th.nn.Linear(hidden_size, n_attributes * n_values),
        )

    def forward(self, x: th.Tensor, input_type: str):
        log_prob = None
        entropy = None
        if input_type == "object":
            x = self.fc1(x)
            if self.training:
                dist = th.distributions.Categorical(logits=x)
                x = dist.sample()
                log_prob = dist.log_prob(x)
                entropy = dist.entropy()
            else:
                x = x.argmax(dim=-1)

            x = th.nn.functional.one_hot(x, self.vocab_size).float()
        elif input_type == "message":
            x = self.fc2(x)
            batch_size = x.shape[0]
            x = x.view(batch_size * self.n_attributes, self.n_values)

        return x, self.Auxiliary(log_prob, entropy)


class SequenceModel(th.nn.Module):
    Auxiliary = namedtuple("Auxiliary", ["log_prob", "entropy"])

    def __init__(
        self,
        n_attributes: int,
        n_values: int,
        vocab_size: int,
        max_len: int = 5,
        hidden_size: int = 64,
        rnn_type: str = "rnn",
        n_layers: int = 2,
        embed_size: int = 32,
    ):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.embed_size = embed_size

        self.object_to_hiddens = th.nn.ModuleList(
            [
                th.nn.Sequential(
                    th.nn.Linear(n_attributes * n_values, hidden_size),
                    th.nn.ReLU(),
                    th.nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(n_layers)
            ]
        )
        self.hidden_to_object = th.nn.Sequential(
            th.nn.Linear(hidden_size, n_attributes * n_values)
        )
        self.hidden_to_logit = th.nn.Sequential(
            th.nn.Linear(hidden_size, vocab_size), th.nn.LogSoftmax(dim=-1)
        )
        self.embedding = th.nn.Embedding(vocab_size, embed_size)
        self.sos_embedding = th.nn.Parameter(th.zeros(embed_size))
        rnn_type = rnn_type.lower()
        rnn_type = {"rnn": th.nn.RNN, "lstm": th.nn.LSTM, "gru": th.nn.GRU}[rnn_type]
        self.rnn = rnn_type(embed_size, hidden_size, n_layers, batch_first=True)

    def forward(self, x: th.Tensor, input_type: str):
        if input_type == "object":
            return self.object_to_message(x)
        if input_type == "message":
            return self.message_to_object(x)

    def object_to_message(self, x: th.Tensor):
        h = th.stack([model(x) for model in self.object_to_hiddens])

        if self.rnn_type == "lstm":
            c = th.zeros_like(h)

        i = self.sos_embedding.repeat(x.size(0), 1)

        sequence = []
        logits = []
        entropy = []

        for _ in range(self.max_len):
            i = i.unsqueeze(1)
            if self.rnn_type == "lstm":
                y, (h, c) = self.rnn(i, (h, c))
            else:
                y, h = self.rnn(i, h)

            logit = self.hidden_to_logit(y.squeeze(1))
            distr = Categorical(logits=logit)

            if self.training:
                x = distr.sample()
            else:
                x = logit.argmax(dim=-1)

            i = self.embedding(x)
            sequence.append(x)
            logits.append(distr.log_prob(x))
            entropy.append(distr.entropy())

        sequence = th.stack(sequence).permute(1, 0)
        logits = th.stack(logits).permute(1, 0)
        entropy = th.stack(entropy).permute(1, 0)

        zeros = th.zeros((sequence.size(0), 1)).to(sequence.device)

        sequence = th.cat([sequence, zeros.long()], dim=1)
        logits = th.cat([logits, zeros], dim=1)
        entropy = th.cat([entropy, zeros], dim=1)

        lengths = util.find_length(sequence)
        max_len = sequence.size(1)
        mask_eos = (
            1
            - th.cumsum(
                th.nn.functional.one_hot(lengths.long(), num_classes=max_len + 1),
                dim=1,
            )[:, :-1]
        )

        sequence = sequence * mask_eos
        logits = (logits * mask_eos).sum(dim=1)
        entropy = (entropy * mask_eos).sum(dim=1) / lengths.float()

        return sequence, self.Auxiliary(logits, entropy)

    def message_to_object(self, x: th.Tensor):
        x = self.embedding(x)
        _, h = self.rnn(x)
        if isinstance(self.rnn, th.nn.LSTM):
            h, _ = h
        x = h[-1]
        x = self.hidden_to_object(x)

        logits = th.ones_like(x)
        entropy = th.zeros_like(x)

        return x, self.Auxiliary(logits, entropy)
