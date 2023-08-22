from dataclasses import dataclass
from typing import Any, Mapping

from ...module import IntSequenceTransformerDecoder, IntSequenceTransformerEncoder
from .agent import Decoder, Encoder
from .train import train


@dataclass
class ConfigTransformer:
    zoo: str

    n_epochs: int
    batch_size: int
    device: str
    seed: int
    wandb_project: str
    use_tqdm: bool

    lr: float
    length: int
    values: int

    metrics_interval: int

    latent_dim: int
    encoder_embed_dim: int
    encoder_n_heads: int
    encoder_ff_dim: int
    encoder_dropout: float
    encoder_activation: str
    encoder_layer_norm_eps: float
    encoder_norm_first: bool
    encoder_n_layers: int
    decoder_embed_dim: int
    decoder_n_heads: int
    decoder_ff_dim: int
    decoder_dropout: float
    decoder_activation: str
    decoder_layer_norm_eps: float
    decoder_norm_first: bool
    decoder_n_layers: int


def train_transformer(config: Mapping[str, Any]) -> None:
    cfg = ConfigTransformer(
        **{k: config[k] for k in ConfigTransformer.__dataclass_fields__}
    )
    encoder = Encoder(
        IntSequenceTransformerEncoder(
            vocab_size=cfg.values,
            output_dim=cfg.latent_dim,
            embed_dim=cfg.encoder_embed_dim,
            n_heads=cfg.encoder_n_heads,
            ff_dim=cfg.encoder_ff_dim,
            dropout=cfg.encoder_dropout,
            activation=cfg.encoder_activation,
            layer_norm_eps=cfg.encoder_layer_norm_eps,
            norm_first=cfg.encoder_norm_first,
            n_layers=cfg.encoder_n_layers,
        )
    )
    decoder = Decoder(
        IntSequenceTransformerDecoder(
            input_dim=cfg.latent_dim,
            max_len=cfg.length,
            vocab_size=cfg.values,
            embed_dim=cfg.decoder_embed_dim,
            n_heads=cfg.decoder_n_heads,
            ff_dim=cfg.decoder_ff_dim,
            dropout=cfg.decoder_dropout,
            activation=cfg.decoder_activation,
            layer_norm_eps=cfg.decoder_layer_norm_eps,
            norm_first=cfg.decoder_norm_first,
            n_layers=cfg.decoder_n_layers,
        )
    )
    train(encoder, decoder, cfg.__dict__)
