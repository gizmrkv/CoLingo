from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Set

from ...game import ReconstructionNetworkGame
from ...module import MLPDecoder, MLPEncoder, RNNDecoder, RNNEncoder
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
    object_values: int
    message_max_len: int
    message_vocab_size: int

    entropy_weight: float
    length_weight: float

    metrics_interval: int
    topsim_interval: int
    language_log_interval: int
    acc_heatmap_interval: int
    lansim_interval: int

    decoder_ae: bool

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

    object_encoder = MLPEncoder(
        max_len=cfg.object_length,
        vocab_size=cfg.object_values,
        output_dim=cfg.latent_dim,
        embed_dim=cfg.object_encoder_embed_dim,
        hidden_dim=cfg.object_encoder_hidden_dim,
        n_layers=cfg.object_encoder_n_layers,
        activation=cfg.object_encoder_activation,
    )
    object_decoder = MLPDecoder(
        input_dim=cfg.latent_dim,
        max_len=cfg.object_length,
        vocab_size=cfg.object_values,
        hidden_dim=cfg.object_decoder_hidden_dim,
        n_layers=cfg.object_decoder_n_layers,
        activation=cfg.object_decoder_activation,
    )
    message_encoder = RNNEncoder(
        vocab_size=cfg.message_vocab_size,
        output_dim=cfg.latent_dim,
        embed_dim=cfg.message_encoder_embed_dim,
        hidden_dim=cfg.message_encoder_hidden_dim,
        rnn_type=cfg.message_encoder_rnn_type,
        n_layers=cfg.message_encoder_n_layers,
    )
    message_decoder = RNNDecoder(
        input_dim=cfg.latent_dim,
        max_len=cfg.message_max_len,
        vocab_size=cfg.message_vocab_size,
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
    elif cfg.network_type == "oneway":
        for i in range(cfg.n_agents - 1):
            adj[agent_name(i)].add(agent_name(i + 1))
    elif cfg.network_type == "ring":
        for i in range(cfg.n_agents):
            adj[agent_name(i)].add(agent_name((i + 1) % cfg.n_agents))

    game = ReconstructionNetworkGame(agents, adj)

    train(agents, game, cfg.__dict__)
