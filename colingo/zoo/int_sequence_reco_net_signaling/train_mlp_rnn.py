from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Set

from ...game import ReconstructionNetworkGame
from ...module import (
    IntSequenceMLPDecoder,
    IntSequenceMLPEncoder,
    IntSequenceRNNDecoder,
    IntSequenceRNNEncoder,
)
from .agent import Agent
from .train import train


@dataclass
class ConfigMLPRNN:
    zoo: str

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

    n_agents: int
    network_type: str

    latent_dim: int

    object_encoder_embed_dim: int
    object_encoder_hidden_dim: int
    object_encoder_n_layers: int
    object_encoder_activation: str

    object_decoder_hidden_dim: int
    object_decoder_n_layers: int
    object_decoder_activation: str

    message_encoder_hidden_dim: int
    message_encoder_embed_dim: int
    message_encoder_rnn_type: str
    message_encoder_n_layers: int

    message_decoder_hidden_dim: int
    message_decoder_embed_dim: int
    message_decoder_rnn_type: str
    message_decoder_n_layers: int


def train_mlp_rnn(config: Mapping[str, Any]) -> None:
    cfg = ConfigMLPRNN(**{k: config[k] for k in ConfigMLPRNN.__dataclass_fields__})

    object_encoder = IntSequenceMLPEncoder(
        length=cfg.object_length,
        n_values=cfg.object_n_values,
        output_dim=cfg.latent_dim,
        embed_dim=cfg.object_encoder_embed_dim,
        hidden_dim=cfg.object_encoder_hidden_dim,
        n_layers=cfg.object_encoder_n_layers,
        activation=cfg.object_encoder_activation,
    )
    object_decoder = IntSequenceMLPDecoder(
        input_dim=cfg.latent_dim,
        length=cfg.object_length,
        n_values=cfg.object_n_values,
        hidden_dim=cfg.object_decoder_hidden_dim,
        n_layers=cfg.object_decoder_n_layers,
        activation=cfg.object_decoder_activation,
    )
    message_encoder = IntSequenceRNNEncoder(
        n_values=cfg.message_n_values,
        output_dim=cfg.latent_dim,
        embed_dim=cfg.message_encoder_embed_dim,
        hidden_dim=cfg.message_encoder_hidden_dim,
        rnn_type=cfg.message_encoder_rnn_type,
        n_layers=cfg.message_encoder_n_layers,
    )
    message_decoder = IntSequenceRNNDecoder(
        input_dim=cfg.latent_dim,
        length=cfg.message_length,
        n_values=cfg.message_n_values,
        embed_dim=cfg.message_decoder_embed_dim,
        hidden_dim=cfg.message_decoder_hidden_dim,
        rnn_type=cfg.message_decoder_rnn_type,
        n_layers=cfg.message_decoder_n_layers,
    )

    def agent_name(i: int) -> str:
        return f"A{i}"

    agents = {
        agent_name(i): Agent(
            object_encoder=deepcopy(object_encoder),
            object_decoder=deepcopy(object_decoder),
            message_encoder=deepcopy(message_encoder),
            message_decoder=deepcopy(message_decoder),
        )
        for i in range(cfg.n_agents)
    }

    adj: Dict[str, Set[str]] = {k: set() for k in agents}

    if cfg.network_type == "linear":
        for i in range(cfg.n_agents - 1):
            adj[agent_name(i)].add(agent_name(i + 1))
            adj[agent_name(i + 1)].add(agent_name(i))

    game = ReconstructionNetworkGame(agents, adj)

    train(agents, game, cfg.__dict__)
