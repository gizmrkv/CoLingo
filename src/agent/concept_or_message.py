import torch as th

from ..model.concept import ConceptDecoder, ConceptEncoder
from ..model.message import MessageDecoder, MessageEncoder


class ConceptOrMessageAgent(th.nn.Module):
    def __init__(
        self,
        n_attributes: int,
        n_values: int,
        max_len: int,
        vocab_size: int,
        internal_dim: int,
        concept_embed_dim: int,
        concept_hidden_dim: int,
        message_embed_dim: int,
        rnn_type: str,
        message_hidden_dim: int,
        n_layers: int,
        share_message_embedding: bool = False,
    ):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.internal_dim = internal_dim
        self.concept_embed_dim = concept_embed_dim
        self.concept_hidden_dim = concept_hidden_dim
        self.message_embed_dim = message_embed_dim
        self.rnn_type = rnn_type
        self.message_hidden_dim = message_hidden_dim
        self.n_layers = n_layers
        self.share_message_embedding = share_message_embedding

        if share_message_embedding:
            self.msg_embed = th.nn.Embedding(vocab_size, message_embed_dim)
        else:
            self.msg_embed = None

        self.concept_encoder = ConceptEncoder(
            n_attributes=n_attributes,
            n_values=n_values,
            embed_dim=internal_dim,
            concept_embed_dim=concept_embed_dim,
        )
        self.concept_decoder = ConceptDecoder(
            n_attributes=n_attributes,
            n_values=n_values,
            embed_dim=internal_dim,
            hidden_dim=concept_hidden_dim,
        )
        self.message_encoder = MessageEncoder(
            max_len=max_len,
            vocab_size=vocab_size,
            embed_dim=internal_dim,
            rnn_type=rnn_type,
            hidden_dim=message_hidden_dim,
            n_layers=n_layers,
            message_embed_dim=message_embed_dim,
            message_embed=self.msg_embed,
        )
        self.message_decoder = MessageDecoder(
            max_len=max_len,
            vocab_size=vocab_size,
            embed_dim=internal_dim,
            rnn_type=rnn_type,
            hidden_dim=message_hidden_dim,
            n_layers=n_layers,
            message_embed_dim=message_embed_dim,
            message_embed=self.msg_embed,
        )

    def forward(
        self,
        input: th.Tensor | None = None,
        message: th.Tensor | None = None,
        hidden: th.Tensor | None = None,
        command: str | None = None,
    ):
        if command == "input":
            match (input, message):
                case (input, None):
                    return self.concept_encoder(input)
                case (None, message):
                    return self.message_encoder(message)
                case (_, _):
                    raise ValueError("input and message cannot both be provided")

        match command:
            case "output":
                return self.concept_decoder(hidden)
            case "message":
                return self.message_decoder(hidden)
            case _:
                raise ValueError(f"unknown command {command}")
