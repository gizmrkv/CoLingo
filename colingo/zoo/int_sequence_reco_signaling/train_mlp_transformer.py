from dataclasses import dataclass
from typing import Any

from ...module import (
    IntSequenceMLPDecoder,
    IntSequenceMLPEncoder,
    IntSequenceTransformerDecoder,
    IntSequenceTransformerEncoder,
)
from .agent import Decoder, Encoder
from .train import train


@dataclass
class ConfigMLPTransformer:
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


def train_mlp_transformer(config: dict[str, Any]) -> None:
    cfg = ConfigMLPTransformer(
        **{k: config[k] for k in ConfigMLPTransformer.__dataclass_fields__}
    )

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
        message_decoder=IntSequenceTransformerDecoder(
            input_dim=cfg.latent_dim,
            length=cfg.message_length,
            n_values=cfg.message_n_values,
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
            n_values=cfg.message_n_values,
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
