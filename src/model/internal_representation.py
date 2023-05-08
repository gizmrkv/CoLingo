from collections import namedtuple

import torch as th
from torch.distributions import Categorical

from ..core import util


class ConceptEncoder(th.nn.Module):
    def __init__(
        self,
        n_attributes: int,
        n_values: int,
        internal_size: int,
        embed_size: int,
        hidden_size: int,
    ):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.internal_size = internal_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = th.nn.ModuleList(
            [th.nn.Embedding(n_values, embed_size) for _ in range(n_attributes)]
        )
        self.fc = th.nn.Sequential(
            th.nn.LayerNorm(embed_size * n_attributes),
            th.nn.Linear(embed_size * n_attributes, hidden_size),
            th.nn.LayerNorm(hidden_size),
            th.nn.ReLU(),
            th.nn.Linear(hidden_size, internal_size),
        )

    def forward(self, x: th.Tensor):
        x = th.cat(
            [self.embedding[i](x[:, i]) for i in range(self.n_attributes)],
            dim=1,
        )
        x = self.fc(x)
        return x


class ConceptDecoder(th.nn.Module):
    def __init__(
        self,
        n_attributes: int,
        n_values: int,
        internal_size: int,
        hidden_size: int,
    ):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.internal_size = internal_size
        self.hidden_size = hidden_size

        self.fc = th.nn.Sequential(
            th.nn.LayerNorm(internal_size),
            th.nn.Linear(internal_size, n_attributes * n_values),
        )

    def forward(self, x: th.Tensor):
        x = self.fc(x)

        logits = th.zeros_like(x).sum(dim=1)
        entropy = th.zeros_like(x).sum(dim=1)

        auxiliary = {"logprob": logits, "entropy": entropy}

        return x, auxiliary


class MessageEncoder(th.nn.Module):
    def __init__(
        self,
        max_len: int,
        vocab_size: int,
        internal_size: int,
        embed_size: int,
        hidden_size: int,
        rnn_type: str,
        n_layers: int,
    ):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.internal_size = internal_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.n_layers = n_layers

        self.embedding = th.nn.Embedding(vocab_size, embed_size)
        rnn_type = rnn_type.lower()
        rnn_type = {"rnn": th.nn.RNN, "lstm": th.nn.LSTM, "gru": th.nn.GRU}[rnn_type]
        self.rnn = rnn_type(embed_size, hidden_size, n_layers, batch_first=True)
        self.fc = th.nn.Linear(hidden_size, internal_size)

    def forward(self, x: th.Tensor):
        x = self.embedding(x)
        x = th.nn.functional.layer_norm(x, x.size()[1:])
        _, h = self.rnn(x)
        if isinstance(self.rnn, th.nn.LSTM):
            h, _ = h
        h = h[-1]
        x = self.fc(h)

        return x


class MessageDecoder(th.nn.Module):
    def __init__(
        self,
        max_len: int,
        vocab_size: int,
        internal_size: int,
        embed_size: int,
        hidden_size: int,
        rnn_type: str,
        n_layers: int,
    ):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.internal_size = internal_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.n_layers = n_layers

        self.internal2hiddens = th.nn.ModuleList(
            [
                th.nn.Sequential(
                    th.nn.LayerNorm(internal_size),
                    th.nn.Linear(internal_size, hidden_size),
                    th.nn.LayerNorm(hidden_size),
                    th.nn.ReLU(),
                    th.nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(n_layers)
            ]
        )

        self.embedding = th.nn.Embedding(vocab_size, embed_size)
        self.sos_embedding = th.nn.Parameter(th.zeros(embed_size))

        self.hidden2vocab = th.nn.Sequential(
            th.nn.Linear(hidden_size, vocab_size), th.nn.LogSoftmax(dim=-1)
        )

        rnn_type = rnn_type.lower()
        rnn_type = {"rnn": th.nn.RNN, "lstm": th.nn.LSTM, "gru": th.nn.GRU}[rnn_type]
        self.rnn = rnn_type(embed_size, hidden_size, n_layers, batch_first=True)

    def forward(self, x: th.Tensor):
        h = th.stack([model(x) for model in self.internal2hiddens])
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

            logit = self.hidden2vocab(y.squeeze(1))
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


class InternalRepresentaionModel(th.nn.Module):
    def __init__(
        self,
        n_attributes: int,
        n_values: int,
        max_len: int,
        vocab_size: int,
        internal_size: int,
        embed_size: int,
        hidden_size: int,
        rnn_type: str,
        n_layers: int,
    ):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.internal_size = internal_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.n_layers = n_layers

        self.concept_encoder = ConceptEncoder(
            n_attributes,
            n_values,
            internal_size,
            embed_size,
            hidden_size,
        )
        self.concept_decoder = ConceptDecoder(
            n_attributes,
            n_values,
            internal_size,
            hidden_size,
        )
        self.message_encoder = MessageEncoder(
            max_len,
            vocab_size,
            internal_size,
            embed_size,
            hidden_size,
            rnn_type,
            n_layers,
        )
        self.message_decoder = MessageDecoder(
            max_len,
            vocab_size,
            internal_size,
            embed_size,
            hidden_size,
            rnn_type,
            n_layers,
        )

    def forward(self, x: th.Tensor, role: str):
        if role == "sender":
            internal = self.concept_encoder(x)
            return self.message_decoder(internal)

        if role == "receiver":
            internal = self.message_encoder(x)
            return self.concept_decoder(internal)
