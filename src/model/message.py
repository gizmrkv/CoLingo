import torch as th
from torch.distributions import Categorical
from torchtyping import TensorType


class MessageEncoder(th.nn.Module):
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

    def forward(self, x: TensorType["batch", "max_len", int]):
        x = self.msg_embed(x)
        _, h = self.rnn(x)
        if isinstance(self.rnn, th.nn.LSTM):
            h, _ = h
        h = h[-1]
        h = self.embed(h)
        return h


class MessageDecoder(th.nn.Module):
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

    def forward(self, x: TensorType["batch", "internal_size", float]):
        h = th.stack([e2h(x) for e2h in self.embed2hiddens])

        if isinstance(self.rnn, th.nn.LSTM):
            c = th.zeros_like(h)

        i = self.sos_embed.repeat(x.size(0), 1)

        message = []
        log_prob = []
        entropy = []

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
            message.append(x)
            log_prob.append(distr.log_prob(x))
            entropy.append(distr.entropy())

        message = th.stack(message).permute(1, 0)
        log_prob = th.stack(log_prob).permute(1, 0)
        entropy = th.stack(entropy).permute(1, 0)

        zeros = th.zeros((message.size(0), 1)).to(message.device)

        message = th.cat([message, zeros.long()], dim=1)
        log_prob = th.cat([log_prob, zeros], dim=1)
        entropy = th.cat([entropy, zeros], dim=1)

        length = message.argmin(dim=1) + 1
        mask_eos = th.arange(self.max_len + 1).expand(message.size(0), -1)
        mask_eos = mask_eos.to(length.device) < length.unsqueeze(1)

        message = message * mask_eos
        log_prob = (log_prob * mask_eos).sum(dim=1)
        entropy = (entropy * mask_eos).sum(dim=1) / length.float()

        return message, log_prob, entropy, length
