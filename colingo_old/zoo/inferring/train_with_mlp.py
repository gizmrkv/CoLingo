from dataclasses import dataclass
from typing import Literal

from ...module import MLPDecoder, MLPEncoder
from .agent import Agent
from .train import Config, train


@dataclass
class ConfigWithMLP:
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
    encoder_hidden_dim: int
    encoder_n_layers: int
    encoder_activation: str

    # decoder config
    decoder_hidden_dim: int
    decoder_n_layers: int
    decoder_activation: str

    # optional config
    use_reinforce: bool = False
    baseline: Literal["batch_mean"] = "batch_mean"
    entropy_weight: float = 0.0


def train_with_mlp(cfg: ConfigWithMLP) -> None:
    encoder = MLPEncoder(
        length=cfg.length,
        n_values=cfg.n_values,
        output_dim=cfg.latent_dim,
        embed_dim=cfg.encoder_embed_dim,
        hidden_dim=cfg.encoder_hidden_dim,
        n_layers=cfg.encoder_n_layers,
        activation=cfg.encoder_activation,
    )
    decoder = MLPDecoder(
        length=cfg.length,
        n_values=cfg.n_values,
        input_dim=cfg.latent_dim,
        hidden_dim=cfg.decoder_hidden_dim,
        n_layers=cfg.decoder_n_layers,
        activation=cfg.decoder_activation,
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
