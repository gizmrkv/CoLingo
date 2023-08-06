from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch.nn as nn

from ...module import MLPDecoder, MLPEncoder, RNNDecoder, RNNEncoder
from .agent import Agent
from .train import Config, train


@dataclass
class ConfigWithMLPRNN:
    # common config
    latent_dim: int
    n_agents: int

    # mlp encoder config
    mlp_encoder_embed_dim: int
    mlp_encoder_hidden_dim: int
    mlp_encoder_n_layers: int
    mlp_encoder_activation: str

    # mlp decoder config
    mlp_decoder_hidden_dim: int
    mlp_decoder_n_layers: int
    mlp_decoder_activation: str

    # encoder config
    rnn_encoder_hidden_dim: int
    rnn_encoder_embed_dim: int
    rnn_encoder_rnn_type: str
    rnn_encoder_n_layers: int

    # decoder config
    rnn_decoder_hidden_dim: int
    rnn_decoder_embed_dim: int
    rnn_decoder_rnn_type: str
    rnn_decoder_n_layers: int

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

    # optional config
    seed: int | None = None

    run_sender_output: bool = False
    run_receiver_send: bool = False
    run_sender_auto_encoding: bool = False
    run_receiver_auto_encoding: bool = False

    receiver_loss_weight: float = 1.0

    sender_loss_weight: float = 1.0
    baseline: str = "batch_mean"
    length_baseline: str = "batch_mean"
    entropy_weight: float = 0.0
    length_weight: float = 0.0

    roce_loss_weight: float | None = None
    raece_loss_weight: float | None = None
    rmce_loss_weight: float | None = None
    smrf_loss_weight: float | None = None
    saece_loss_weight: float | None = None
    soce_loss_weight: float | None = None
    saecel_loss_weight: float | None = None

    eval_interval: int = 10
    acc_heatmap_interval: int = 5
    topsim_interval: int = 5
    lansim_interval: int = 50
    lansim_heatmap_interval: int = 1
    language_save_interval: int = 50


def train_with_mlp_rnn(cfg: ConfigWithMLPRNN) -> None:
    object_encoder = MLPEncoder(
        length=cfg.object_length,
        n_values=cfg.object_n_values,
        output_dim=cfg.latent_dim,
        embed_dim=cfg.mlp_encoder_embed_dim,
        hidden_dim=cfg.mlp_encoder_hidden_dim,
        n_layers=cfg.mlp_encoder_n_layers,
        activation=cfg.mlp_encoder_activation,
    )
    object_decoder = MLPDecoder(
        length=cfg.object_length,
        n_values=cfg.object_n_values,
        input_dim=cfg.latent_dim,
        hidden_dim=cfg.mlp_decoder_hidden_dim,
        n_layers=cfg.mlp_decoder_n_layers,
        activation=cfg.mlp_decoder_activation,
    )
    message_encoder = RNNEncoder(
        n_values=cfg.message_n_values,
        output_dim=cfg.latent_dim,
        hidden_dim=cfg.rnn_encoder_hidden_dim,
        embed_dim=cfg.rnn_encoder_embed_dim,
        rnn_type=cfg.rnn_encoder_rnn_type,
        n_layers=cfg.rnn_encoder_n_layers,
    )
    message_decoder = RNNDecoder(
        input_dim=cfg.latent_dim,
        length=cfg.message_length,
        n_values=cfg.message_n_values,
        hidden_dim=cfg.rnn_decoder_hidden_dim,
        embed_dim=cfg.rnn_decoder_embed_dim,
        rnn_type=cfg.rnn_decoder_rnn_type,
        n_layers=cfg.rnn_decoder_n_layers,
    )
    shared = nn.LayerNorm(cfg.latent_dim)
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

    adj = {k: v for k, v in adj.items() if len(v) > 0}

    train(
        agents,
        adj,
        Config(
            # exp config
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
            device=cfg.device,
            zoo_name=cfg.zoo_name,
            wandb_project=cfg.wandb_project,
            use_tqdm=cfg.use_tqdm,
            lr=cfg.lr,
            object_length=cfg.object_length,
            object_n_values=cfg.object_n_values,
            message_length=cfg.message_length,
            message_n_values=cfg.message_n_values,
            seed=cfg.seed,
            run_sender_output=cfg.run_sender_output,
            run_receiver_send=cfg.run_receiver_send,
            run_sender_auto_encoding=cfg.run_sender_auto_encoding,
            run_receiver_auto_encoding=cfg.run_receiver_auto_encoding,
            baseline=cfg.baseline,
            length_baseline=cfg.length_baseline,
            entropy_weight=cfg.entropy_weight,
            length_weight=cfg.length_weight,
            roce_loss_weight=cfg.roce_loss_weight,
            raece_loss_weight=cfg.raece_loss_weight,
            rmce_loss_weight=cfg.rmce_loss_weight,
            smrf_loss_weight=cfg.smrf_loss_weight,
            saece_loss_weight=cfg.saece_loss_weight,
            soce_loss_weight=cfg.soce_loss_weight,
            saecel_loss_weight=cfg.saecel_loss_weight,
            eval_interval=cfg.eval_interval,
            acc_heatmap_interval=cfg.acc_heatmap_interval,
            topsim_interval=cfg.topsim_interval,
            lansim_interval=cfg.lansim_interval,
            lansim_heatmap_interval=cfg.lansim_heatmap_interval,
            language_save_interval=cfg.language_save_interval,
        ),
    )
