from dataclasses import dataclass
from typing import Literal

from ...module import TransformerDecoder, TransformerEncoder
from .agent import Agent
from .train import Config, train


@dataclass
class ConfigWithTransformer:
    # exp config
    n_epochs: int
    batch_size: int
    seed: int
    device: str
    zoo_name: str
    wandb_project: str
    use_tqdm: bool

    # common config
    lr: float
    length: int
    n_values: int
    latent_dim: int

    # encoder config
    encoder_embed_dim: int
    encoder_n_heads: int
    encoder_ff_dim: int
    encoder_dropout: float
    encoder_activation: str
    encoder_layer_norm_eps: float
    encoder_n_layers: int

    # decoder config
    decoder_embed_dim: int
    decoder_n_heads: int
    decoder_ff_dim: int
    decoder_dropout: float
    decoder_activation: str
    decoder_layer_norm_eps: float
    decoder_n_layers: int

    # optional config
    use_reinforce: bool = False
    baseline: Literal["batch_mean"] = "batch_mean"
    entropy_weight: float = 0.0


def train_with_transformer(cfg: ConfigWithTransformer) -> None:
    encoder = TransformerEncoder(
        n_values=cfg.n_values,
        output_dim=cfg.latent_dim,
        embed_dim=cfg.encoder_embed_dim,
        n_heads=cfg.encoder_n_heads,
        ff_dim=cfg.encoder_ff_dim,
        dropout=cfg.encoder_dropout,
        activation=cfg.encoder_activation,
        layer_norm_eps=cfg.encoder_layer_norm_eps,
        batch_first=True,
        n_layers=cfg.encoder_n_layers,
    )
    decoder = TransformerDecoder(
        input_dim=cfg.latent_dim,
        length=cfg.length,
        n_values=cfg.n_values,
        embed_dim=cfg.decoder_embed_dim,
        n_heads=cfg.decoder_n_heads,
        ff_dim=cfg.decoder_ff_dim,
        dropout=cfg.decoder_dropout,
        activation=cfg.decoder_activation,
        layer_norm_eps=cfg.decoder_layer_norm_eps,
        batch_first=True,
        n_layers=cfg.decoder_n_layers,
    )
    agent = Agent(encoder, decoder)
    train(
        agent,
        Config(
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            device=cfg.device,
            zoo_name=cfg.zoo_name,
            wandb_project=cfg.wandb_project,
            use_tqdm=cfg.use_tqdm,
            lr=cfg.lr,
            length=cfg.length,
            n_values=cfg.n_values,
            use_reinforce=cfg.use_reinforce,
            baseline=cfg.baseline,
            entropy_weight=cfg.entropy_weight,
        ),
    )
