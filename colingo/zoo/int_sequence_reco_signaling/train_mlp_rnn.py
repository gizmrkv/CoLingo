from dataclasses import dataclass
from typing import Any

from ...module import (
    IntSequenceMLPDecoder,
    IntSequenceMLPEncoder,
    IntSequenceRNNDecoder,
    IntSequenceRNNEncoder,
)
from .agent import Decoder, Encoder
from .train import train


@dataclass
class ConfigMLPRNN:
    n_epochs: int
    batch_size: int
    device: str
    seed: int
    wandb_project: str
    use_tqdm: bool

    lr: float
    object_length: int
    object_n_values: int
    message_length: int
    message_n_values: int

    entropy_weight: float
    length_weight: float

    metrics_interval: int
    topsim_interval: int
    language_log_interval: int

    latent_dim: int

    object_encoder_embed_dim: int
    object_encoder_hidden_dim: int
    object_encoder_n_layers: int
    object_encoder_activation: str

    message_decoder_hidden_dim: int
    message_decoder_embed_dim: int
    message_decoder_rnn_type: str
    message_decoder_n_layers: int

    message_encoder_hidden_dim: int
    message_encoder_embed_dim: int
    message_encoder_rnn_type: str
    message_encoder_n_layers: int

    object_decoder_hidden_dim: int
    object_decoder_n_layers: int
    object_decoder_activation: str


def train_mlp_rnn(config: dict[str, Any]) -> None:
    cfg = ConfigMLPRNN(**{k: config[k] for k in ConfigMLPRNN.__dataclass_fields__})

    encoder = Encoder(
        object_encoder=IntSequenceMLPEncoder(
            length=cfg.object_length,
            n_values=cfg.object_n_values,
            output_dim=cfg.latent_dim,
            embed_dim=cfg.object_encoder_embed_dim,
            hidden_dim=cfg.object_encoder_hidden_dim,
            n_layers=cfg.object_encoder_n_layers,
            activation=cfg.object_encoder_activation,
        ),
        message_decoder=IntSequenceRNNDecoder(
            input_dim=cfg.latent_dim,
            length=cfg.message_length,
            n_values=cfg.message_n_values,
            embed_dim=cfg.message_decoder_embed_dim,
            hidden_dim=cfg.message_decoder_hidden_dim,
            rnn_type=cfg.message_decoder_rnn_type,
            n_layers=cfg.message_decoder_n_layers,
        ),
    )
    decoder = Decoder(
        message_encoder=IntSequenceRNNEncoder(
            n_values=cfg.message_n_values,
            output_dim=cfg.latent_dim,
            embed_dim=cfg.message_encoder_embed_dim,
            hidden_dim=cfg.message_encoder_hidden_dim,
            rnn_type=cfg.message_encoder_rnn_type,
            n_layers=cfg.message_encoder_n_layers,
        ),
        object_decoder=IntSequenceMLPDecoder(
            input_dim=cfg.latent_dim,
            length=cfg.object_length,
            n_values=cfg.object_n_values,
            hidden_dim=cfg.object_decoder_hidden_dim,
            n_layers=cfg.object_decoder_n_layers,
            activation=cfg.object_decoder_activation,
        ),
    )
    train(encoder, decoder, cfg.__dict__)
