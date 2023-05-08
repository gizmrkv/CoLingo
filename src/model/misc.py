from collections import namedtuple

import torch as th
from torch.distributions import Categorical

from ..core import util


class OnehotConceptSymbolMessageModel(th.nn.Module):
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

    def forward(self, x: th.Tensor, role: str):
        log_prob = None
        entropy = None
        if role == "sender":
            x = self.fc1(x)
            if self.training:
                dist = th.distributions.Categorical(logits=x)
                x = dist.sample()
                log_prob = dist.log_prob(x)
                entropy = dist.entropy()
            else:
                x = x.argmax(dim=-1)

            x = th.nn.functional.one_hot(x, self.vocab_size).float()
        elif role == "receiver":
            x = self.fc2(x)
            batch_size = x.shape[0]
            x = x.view(batch_size * self.n_attributes, self.n_values)

        return x, self.Auxiliary(log_prob, entropy)


class OnehotConceptSequntialMessageModel(th.nn.Module):
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

    def forward(self, x: th.Tensor, role: str):
        if role == "sender":
            return self.object_to_message(x)
        if role == "receiver":
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

        length = util.find_length(sequence)
        max_len = sequence.size(1)
        mask_eos = (
            1
            - th.cumsum(
                th.nn.functional.one_hot(length.long(), num_classes=max_len + 1),
                dim=1,
            )[:, :-1]
        )

        sequence = sequence * mask_eos
        logits = (logits * mask_eos).sum(dim=1)
        entropy = (entropy * mask_eos).sum(dim=1) / length.float()

        auxiliary = {"logprob": logits, "entropy": entropy, "length": length}

        return sequence, auxiliary

    def message_to_object(self, x: th.Tensor):
        x = self.embedding(x)
        _, h = self.rnn(x)
        if isinstance(self.rnn, th.nn.LSTM):
            h, _ = h
        x = h[-1]
        x = self.hidden_to_object(x)

        logits = th.zeros_like(x).sum(dim=1)
        entropy = th.zeros_like(x).sum(dim=1)

        auxiliary = {"logprob": logits, "entropy": entropy}

        return x, auxiliary


class EmbeddingConceptSequentialMessageModel(th.nn.Module):
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

        self.attributes_embedding = th.nn.ModuleList(
            [th.nn.Embedding(n_values, embed_size) for _ in range(n_attributes)]
        )
        self.object_to_hiddens = th.nn.ModuleList(
            [
                th.nn.Sequential(
                    th.nn.Linear(n_attributes * embed_size, hidden_size),
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

    def forward(self, x: th.Tensor, role: str):
        if role == "sender":
            return self.object_to_message(x)
        if role == "receiver":
            return self.message_to_object(x)

    def object_to_message(self, x: th.Tensor):
        x = th.cat(
            [self.attributes_embedding[i](x[:, i]) for i in range(self.n_attributes)],
            dim=1,
        )
        x = th.nn.functional.layer_norm(x, x.size()[1:])
        h = th.stack([model(x) for model in self.object_to_hiddens])
        h = th.nn.functional.layer_norm(h, h.size()[1:])

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
            logit = th.nn.functional.layer_norm(logit, logit.size()[1:])
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

        length = util.find_length(sequence)
        max_len = sequence.size(1)
        mask_eos = (
            1
            - th.cumsum(
                th.nn.functional.one_hot(length.long(), num_classes=max_len + 1),
                dim=1,
            )[:, :-1]
        )

        sequence = sequence * mask_eos
        logits = (logits * mask_eos).sum(dim=1)
        entropy = (entropy * mask_eos).sum(dim=1) / length.float()

        auxiliary = {"logprob": logits, "entropy": entropy, "length": length}

        return sequence, auxiliary

    def message_to_object(self, x: th.Tensor):
        x = self.embedding(x)
        _, h = self.rnn(x)
        if isinstance(self.rnn, th.nn.LSTM):
            h, _ = h
        x = h[-1]
        x = self.hidden_to_object(x)

        logits = th.zeros_like(x).sum(dim=1)
        entropy = th.zeros_like(x).sum(dim=1)

        auxiliary = {"logprob": logits, "entropy": entropy}

        return x, auxiliary
