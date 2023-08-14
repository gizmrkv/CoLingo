from dataclasses import dataclass
from typing import Any, Mapping

from ...module import IntSequenceMLPDecoder, IntSequenceMLPEncoder
from .agent import Decoder, Encoder
from .train import train


@dataclass
class ConfigMLP:
    zoo: str

    n_epochs: int
    batch_size: int
    device: str
    seed: int
    wandb_project: str
    use_tqdm: bool

    lr: float
    length: int
    n_values: int

    metrics_interval: int

    latent_dim: int
    encoder_embed_dim: int
    encoder_hidden_dim: int
    encoder_n_layers: int
    encoder_activation: str
    decoder_embed_dim: int
    decoder_hidden_dim: int
    decoder_n_layers: int
    decoder_activation: str


def train_mlp(config: Mapping[str, Any]) -> None:
    cfg = ConfigMLP(**{k: config[k] for k in ConfigMLP.__dataclass_fields__})
    encoder = Encoder(
        IntSequenceMLPEncoder(
            length=cfg.length,
            n_values=cfg.n_values,
            output_dim=cfg.latent_dim,
            embed_dim=cfg.encoder_embed_dim,
            hidden_dim=cfg.encoder_hidden_dim,
            n_layers=cfg.encoder_n_layers,
            activation=cfg.encoder_activation,
        )
    )
    decoder = Decoder(
        IntSequenceMLPDecoder(
            input_dim=cfg.latent_dim,
            length=cfg.length,
            n_values=cfg.n_values,
            hidden_dim=cfg.decoder_hidden_dim,
            n_layers=cfg.decoder_n_layers,
            activation=cfg.decoder_activation,
        )
    )
    train(encoder, decoder, cfg.__dict__)
