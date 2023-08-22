from dataclasses import dataclass
from typing import Any, Mapping

from ...module import (
    IntSequenceTransformerDecoder,
    IntSequenceTransformerEncoder,
    MLPDecoder,
    MLPEncoder,
)
from .agent import Decoder, Encoder
from .train import train


@dataclass
class ConfigMLPTransformer:
    zoo: str

    n_epochs: int
    batch_size: int
    device: str
    seed: int
    wandb_project: str
    use_tqdm: bool

    lr: float
    object_length: int
    object_values: int
    message_max_len: int
    message_vocab_size: int

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

    message_decoder_embed_dim: int
    message_decoder_n_heads: int
    message_decoder_ff_dim: int
    message_decoder_dropout: float
    message_decoder_activation: str
    message_decoder_layer_norm_eps: float
    message_decoder_norm_first: bool
    message_decoder_n_layers: int

    message_encoder_embed_dim: int
    message_encoder_n_heads: int
    message_encoder_ff_dim: int
    message_encoder_dropout: float
    message_encoder_activation: str
    message_encoder_layer_norm_eps: float
    message_encoder_norm_first: bool
    message_encoder_n_layers: int

    object_decoder_hidden_dim: int
    object_decoder_n_layers: int
    object_decoder_activation: str


def train_mlp_transformer(config: Mapping[str, Any]) -> None:
    cfg = ConfigMLPTransformer(
        **{k: config[k] for k in ConfigMLPTransformer.__dataclass_fields__}
    )

    encoder = Encoder(
        object_encoder=MLPEncoder(
            max_len=cfg.object_length,
            vocab_size=cfg.object_values,
            output_dim=cfg.latent_dim,
            embed_dim=cfg.object_encoder_embed_dim,
            hidden_dim=cfg.object_encoder_hidden_dim,
            n_layers=cfg.object_encoder_n_layers,
            activation=cfg.object_encoder_activation,
        ),
        message_decoder=IntSequenceTransformerDecoder(
            input_dim=cfg.latent_dim,
            max_len=cfg.message_max_len,
            vocab_size=cfg.message_vocab_size,
            embed_dim=cfg.message_decoder_embed_dim,
            n_heads=cfg.message_decoder_n_heads,
            ff_dim=cfg.message_decoder_ff_dim,
            dropout=cfg.message_decoder_dropout,
            activation=cfg.message_decoder_activation,
            layer_norm_eps=cfg.message_decoder_layer_norm_eps,
            norm_first=cfg.message_decoder_norm_first,
            n_layers=cfg.message_decoder_n_layers,
        ),
    )
    decoder = Decoder(
        message_encoder=IntSequenceTransformerEncoder(
            vocab_size=cfg.message_vocab_size,
            output_dim=cfg.latent_dim,
            embed_dim=cfg.message_encoder_embed_dim,
            n_heads=cfg.message_encoder_n_heads,
            ff_dim=cfg.message_encoder_ff_dim,
            dropout=cfg.message_encoder_dropout,
            activation=cfg.message_encoder_activation,
            layer_norm_eps=cfg.message_encoder_layer_norm_eps,
            norm_first=cfg.message_encoder_norm_first,
            n_layers=cfg.message_encoder_n_layers,
        ),
        object_decoder=MLPDecoder(
            input_dim=cfg.latent_dim,
            max_len=cfg.object_length,
            vocab_size=cfg.object_values,
            hidden_dim=cfg.object_decoder_hidden_dim,
            n_layers=cfg.object_decoder_n_layers,
            activation=cfg.object_decoder_activation,
        ),
    )
    train(encoder, decoder, cfg.__dict__)
