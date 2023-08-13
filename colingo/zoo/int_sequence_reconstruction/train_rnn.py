from dataclasses import dataclass
from typing import Any, Mapping

from ...module import IntSequenceRNNDecoder, IntSequenceRNNEncoder
from .agent import Decoder, Encoder
from .train import train


@dataclass
class ConfigRNN:
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
    encoder_rnn_type: str
    encoder_n_layers: int
    decoder_embed_dim: int
    decoder_hidden_dim: int
    decoder_rnn_type: str
    decoder_n_layers: int


def train_rnn(config: Mapping[str, Any]) -> None:
    cfg = ConfigRNN(**{k: config[k] for k in ConfigRNN.__dataclass_fields__})
    encoder = Encoder(
        IntSequenceRNNEncoder(
            n_values=cfg.n_values,
            output_dim=cfg.latent_dim,
            embed_dim=cfg.encoder_embed_dim,
            hidden_dim=cfg.encoder_hidden_dim,
            rnn_type=cfg.encoder_rnn_type,
            n_layers=cfg.encoder_n_layers,
        )
    )
    decoder = Decoder(
        IntSequenceRNNDecoder(
            input_dim=cfg.latent_dim,
            length=cfg.length,
            n_values=cfg.n_values,
            embed_dim=cfg.decoder_embed_dim,
            hidden_dim=cfg.decoder_hidden_dim,
            rnn_type=cfg.decoder_rnn_type,
            n_layers=cfg.decoder_n_layers,
        )
    )
    train(encoder, decoder, cfg.__dict__)
