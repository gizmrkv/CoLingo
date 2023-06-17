import torch as th
from torch.distributions import Categorical
from torchtyping import TensorType

from ..message import SequenceMessage


class SequenceMessageEncoder(th.nn.Module):
    def __init__(
        self,
        max_len: int,
        vocab_size: int,
        embed_dim: int,
        rnn_type: str,
        hidden_dim: int,
        n_layers: int,
        message_embed_dim: int,
        message_embed: th.nn.Embedding | None = None,
    ):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.message_embed_dim = message_embed_dim

        self.msg_embed = (
            th.nn.Embedding(vocab_size, message_embed_dim)
            if message_embed is None
            else message_embed
        )
        rnn_type = rnn_type.lower()
        rnn_type = {"rnn": th.nn.RNN, "lstm": th.nn.LSTM, "gru": th.nn.GRU}[rnn_type]
        self.rnn = rnn_type(message_embed_dim, hidden_dim, n_layers, batch_first=True)
        self.embed = th.nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: SequenceMessage):
        x = self.msg_embed(x.sequence)
        _, h = self.rnn(x)
        if isinstance(self.rnn, th.nn.LSTM):
            h, _ = h
        h = h[-1]
        h = self.embed(h)
        return h


class SequenceMessageDecoder(th.nn.Module):
    def __init__(
        self,
        max_len: int,
        vocab_size: int,
        embed_dim: int,
        rnn_type: str,
        hidden_dim: int,
        n_layers: int,
        message_embed_dim: int,
        message_embed: th.nn.Embedding | None = None,
    ):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.message_embed_dim = message_embed_dim

        self.embed2hiddens = th.nn.ModuleList(
            [th.nn.Linear(embed_dim, hidden_dim) for _ in range(n_layers)]
        )

        rnn_type = rnn_type.lower()
        rnn_type = {"rnn": th.nn.RNN, "lstm": th.nn.LSTM, "gru": th.nn.GRU}[rnn_type]
        self.rnn = rnn_type(message_embed_dim, hidden_dim, n_layers, batch_first=True)

        self.sos_embed = th.nn.Parameter(th.zeros(message_embed_dim))
        self.hidden2logits = th.nn.Linear(hidden_dim, vocab_size)
        self.msg_embed = (
            th.nn.Embedding(vocab_size, message_embed_dim)
            if message_embed is None
            else message_embed
        )

    def rnn_hidden(self, x: TensorType["batch", "embed_dim", float]):
        h = th.stack([e2h(x) for e2h in self.embed2hiddens])

        if isinstance(self.rnn, th.nn.LSTM):
            h = (h, th.zeros_like(h))

        return h

    def forward(self, x: TensorType["batch", "internal_size", float]):
        h = self.rnn_hidden(x)
        if isinstance(self.rnn, th.nn.LSTM):
            h, c = h

        i = self.sos_embed.repeat(x.size(0), 1)

        sequence = []
        logits = []
        log_probs = []
        entropies = []

        for _ in range(self.max_len):
            i = i.unsqueeze(1)
            if isinstance(self.rnn, th.nn.LSTM):
                y, (h, c) = self.rnn(i, (h, c))
            else:
                y, h = self.rnn(i, h)

            logit = self.hidden2logits(y.squeeze(1))
            distr = Categorical(logits=logit)

            if self.training:
                x = distr.sample()
            else:
                x = logit.argmax(dim=-1)

            i = self.msg_embed(x)
            sequence.append(x)
            logits.append(logit)
            log_probs.append(distr.log_prob(x))
            entropies.append(distr.entropy())

        sequence = th.stack(sequence).permute(1, 0)
        logits = th.stack(logits).permute(1, 0, 2)
        log_probs = th.stack(log_probs).permute(1, 0)
        entropies = th.stack(entropies).permute(1, 0)

        zeros = th.zeros((sequence.size(0), 1)).to(sequence.device)

        sequence = th.cat([sequence, zeros.long()], dim=1)
        log_probs = th.cat([log_probs, zeros], dim=1)
        entropies = th.cat([entropies, zeros], dim=1)

        length = sequence.argmin(dim=1) + 1
        eos_mask = th.arange(self.max_len + 1).expand(sequence.size(0), -1)
        eos_mask = eos_mask.to(length.device) < length.unsqueeze(1)

        sequence = sequence * eos_mask
        log_probs = log_probs * eos_mask
        log_prob = log_probs.sum(dim=1)
        entropies = entropies * eos_mask
        entropy = entropies.sum(dim=1) / length.float()

        return SequenceMessage(
            batch_size=sequence.size(0),
            max_len=self.max_len + 1,
            vocab_size=self.vocab_size,
            sequence=sequence,
            logits=logits,
            log_probs=log_probs,
            log_prob=log_prob,
            entropies=entropies,
            entropy=entropy,
            length=length,
        )
