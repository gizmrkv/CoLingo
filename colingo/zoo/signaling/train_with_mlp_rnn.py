from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Mapping

from ...module import (
    ContMLP,
    DiscSeqMLPDecoder,
    DiscSeqMLPEncoder,
    DiscSeqRNNDecoder,
    DiscSeqRNNEncoder,
)
from .agent import Agent
from .loss import (
    Loss,
    ReceiverAutoEncodingCrossEntropyLoss,
    ReceiverMessageCrossEntropyLoss,
    ReceiverObjectCrossEntropyLoss,
    SenderAutoEncodingCrossEntropyLoss,
    SenderMessageReinforceLoss,
    SenderObjectCrossEntropyLoss,
)
from .train import Config, train


@dataclass
class ConfigWithMLPRNN:
    # exp config
    n_epochs: int
    batch_size: int
    device: str
    zoo_name: str
    wandb_project: str
    use_tqdm: bool

    # common config
    lr: float
    object_length: int
    object_n_values: int
    message_length: int
    message_n_values: int
    latent_dim: int
    n_agents: int

    # mlp encoder config
    mlp_encoder_hidden_dim: int
    mlp_encoder_embed_dim: int
    mlp_encoder_activation: str
    mlp_encoder_use_layer_norm: bool
    mlp_encoder_use_residual: bool
    mlp_encoder_n_blocks: int

    # mlp decoder config
    mlp_decoder_hidden_dim: int
    mlp_decoder_activation: str
    mlp_decoder_use_layer_norm: bool
    mlp_decoder_use_residual: bool
    mlp_decoder_n_blocks: int

    # encoder config
    rnn_encoder_hidden_dim: int
    rnn_encoder_embed_dim: int
    rnn_encoder_rnn_type: Literal["rnn", "lstm", "gru"]
    rnn_encoder_n_layers: int

    # decoder config
    rnn_decoder_hidden_dim: int
    rnn_decoder_embed_dim: int
    rnn_decoder_rnn_type: Literal["rnn", "lstm", "gru"]
    rnn_decoder_n_layers: int

    # optional config
    seed: int | None = None
    use_reinforce: bool = False
    baseline: Literal["batch_mean"] = "batch_mean"
    entropy_weight: float = 0.0
    length_weight: float = 0.0
    sender_loss_weight: float = 1.0
    receiver_loss_weight: float = 1.0


def train_with_mlp_rnn(cfg: ConfigWithMLPRNN) -> None:
    object_encoder = DiscSeqMLPEncoder(
        length=cfg.object_length,
        n_values=cfg.object_n_values,
        output_dim=cfg.latent_dim,
        hidden_dim=cfg.mlp_encoder_hidden_dim,
        embed_dim=cfg.mlp_encoder_embed_dim,
        activation=cfg.mlp_encoder_activation,
        use_layer_norm=cfg.mlp_encoder_use_layer_norm,
        use_residual=cfg.mlp_encoder_use_residual,
        n_blocks=cfg.mlp_encoder_n_blocks,
    )
    object_decoder = DiscSeqMLPDecoder(
        length=cfg.object_length,
        n_values=cfg.object_n_values,
        input_dim=cfg.latent_dim,
        hidden_dim=cfg.mlp_decoder_hidden_dim,
        activation=cfg.mlp_decoder_activation,
        use_layer_norm=cfg.mlp_decoder_use_layer_norm,
        use_residual=cfg.mlp_decoder_use_residual,
        n_blocks=cfg.mlp_decoder_n_blocks,
    )
    message_encoder = DiscSeqRNNEncoder(
        n_values=cfg.message_n_values,
        output_dim=cfg.latent_dim,
        hidden_dim=cfg.rnn_encoder_hidden_dim,
        embed_dim=cfg.rnn_encoder_embed_dim,
        rnn_type=cfg.rnn_encoder_rnn_type,
        n_layers=cfg.rnn_encoder_n_layers,
    )
    message_decoder = DiscSeqRNNDecoder(
        length=cfg.message_length,
        n_values=cfg.message_n_values,
        input_dim=cfg.latent_dim,
        hidden_dim=cfg.rnn_decoder_hidden_dim,
        embed_dim=cfg.rnn_decoder_embed_dim,
        rnn_type=cfg.rnn_decoder_rnn_type,
        n_layers=cfg.rnn_decoder_n_layers,
    )
    shared = ContMLP(cfg.latent_dim, cfg.latent_dim, cfg.latent_dim)
    agents = {
        f"A{i}": Agent(
            object_encoder=deepcopy(object_encoder),
            object_decoder=deepcopy(object_decoder),
            message_encoder=deepcopy(message_encoder),
            message_decoder=deepcopy(message_decoder),
            shared=deepcopy(shared),
        )
        for i in range(cfg.n_agents)
    }

    # adj: dict[str, list[str]] = {name: [name] for name in agents}
    adj: dict[str, list[str]] = {name: [] for name in agents}
    names = list(agents.keys())
    for i in range(len(names) - 1):
        adj[names[i]].append(names[i + 1])
        adj[names[i + 1]].append(names[i])

    train(
        agents,
        adj,
        {},
        Config(
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            device=cfg.device,
            zoo_name=cfg.zoo_name,
            wandb_project=cfg.wandb_project,
            use_tqdm=cfg.use_tqdm,
            lr=cfg.lr,
            object_length=cfg.object_length,
            object_n_values=cfg.object_n_values,
            message_length=cfg.message_length,
            message_n_values=cfg.message_n_values,
            use_reinforce=cfg.use_reinforce,
            baseline=cfg.baseline,
            entropy_weight=cfg.entropy_weight,
            length_weight=cfg.length_weight,
            sender_loss_weight=cfg.sender_loss_weight,
            receiver_loss_weight=cfg.receiver_loss_weight,
        ),
    )
