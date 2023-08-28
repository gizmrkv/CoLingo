from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Set

from ...game import ReconstructionNetworkGame
from ...module import MLPDecoder, MLPEncoder, TransformerDecoder, TransformerEncoder
from .agent import Agent
from .train import train


@dataclass
class ConfigMLPTransformer:
    zoo: str

    n_epochs: int
    batch_size: int
    device: str
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
    network_window: int

    latent_dim: int

    object_encoder_embed_dim: int
    object_encoder_hidden_dim: int
    object_encoder_n_layers: int
    object_encoder_activation: str

    object_decoder_hidden_dim: int
    object_decoder_n_layers: int
    object_decoder_activation: str

    message_encoder_embed_dim: int
    message_encoder_n_heads: int
    message_encoder_ff_dim: int
    message_encoder_dropout: float
    message_encoder_activation: str
    message_encoder_layer_norm_eps: float
    message_encoder_norm_first: bool
    message_encoder_n_layers: int
    message_encoder_is_causal: bool

    message_decoder_embed_dim: int
    message_decoder_n_heads: int
    message_decoder_ff_dim: int
    message_decoder_dropout: float
    message_decoder_activation: str
    message_decoder_layer_norm_eps: float
    message_decoder_norm_first: bool
    message_decoder_n_layers: int


def train_mlp_transformer(config: Mapping[str, Any], log_dir: Path) -> None:
    cfg = ConfigMLPTransformer(
        **{k: config[k] for k in ConfigMLPTransformer.__dataclass_fields__}
    )

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
    message_encoder = TransformerEncoder(
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
        is_causal=cfg.message_encoder_is_causal,
    )
    message_decoder = TransformerDecoder(
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

    assert 1 <= cfg.network_window <= cfg.n_agents

    for i in range(cfg.n_agents):
        for w in range(cfg.network_window):
            adj[agent_name(i)].add(agent_name((i + w + 1) % cfg.n_agents))

    game = ReconstructionNetworkGame(agents, adj)

    train(agents, game, cfg.__dict__, log_dir)
