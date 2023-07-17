from dataclasses import dataclass
from itertools import combinations, permutations, product

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

from ...analysis import language_similarity, topographic_similarity
from ...baseline import BatchMeanBaseline
from ...core import Evaluator, Interval, Runner, Trainer, fix_seed, init_weights
from ...dataset import random_split
from ...game import SignalingGame, SignalingGameResult
from ...logger import EarlyStopper, StepCounter, WandBLogger
from ...loss import DiscSeqReinforceLoss
from ...module import (
    DiscSeqMLPDecoder,
    DiscSeqMLPEncoder,
    DiscSeqRNNDecoder,
    DiscSeqRNNEncoder,
)
from .train import main as train


@dataclass
class Config:
    # exp config
    n_epochs: int
    batch_size: int
    seed: int
    device: str
    exp_name: str
    wandb_project: str
    use_tqdm: bool

    # common config
    lr: float
    n_agents: int
    input_length: int
    input_n_values: int
    message_length: int
    message_n_values: int
    latent_dim: int
    entropy_weight: float
    length_weight: float
    baseline: str
    length_baseline: str

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
    rnn_encoder_rnn_type: str
    rnn_encoder_n_layers: int

    # decoder config
    rnn_decoder_hidden_dim: int
    rnn_decoder_embed_dim: int
    rnn_decoder_rnn_type: str
    rnn_decoder_n_layers: int

    # optional config
    run_sender_output: bool = False
    run_receiver_send: bool = False
    run_receiver_rec: bool = False


class Agent(nn.Module):
    def __init__(
        self,
        input_encoder,
        input_decoder,
        message_encoder,
        message_decoder,
    ):
        super().__init__()
        self.input_encoder = input_encoder
        self.input_decoder = input_decoder
        self.message_encoder = message_encoder
        self.message_decoder = message_decoder

    def forward(self, command: str, input=None, message=None, latent=None):
        match command:
            case "input":
                return self.input_encoder(input)
            case "output":
                return self.input_decoder(latent)
            case "receive":
                return self.message_encoder(message)
            case "send":
                return self.message_decoder(latent)
            case "echo":
                return self.message_decoder(latent, message)
            case _:
                raise ValueError(f"Unknown command: {command}")


class DiscSeqAdapter(nn.Module):
    def __init__(self, module: nn.Module, padding: bool = False, eos: int = 0):
        super().__init__()
        self.module = module
        self.padding = padding
        self.eos = eos

    def forward(self, *args, **kwargs):
        x, logits = self.module(*args, **kwargs)

        distr = Categorical(logits=logits)
        log_prob = distr.log_prob(x)
        entropy = distr.entropy()

        mask = None
        length = None

        if self.padding:
            mask = x == self.eos
            indices = torch.argmax(mask.int(), dim=1)
            no_mask = ~mask.any(dim=1)
            indices[no_mask] = x.shape[1]
            mask = torch.arange(x.shape[1]).expand(x.shape).to(x.device)
            mask = (mask <= indices.unsqueeze(-1)).long()

            length = mask.sum(dim=-1)
            x = x * mask
            log_prob = log_prob * mask
            entropy = entropy * mask

        return x, {
            "log_prob": log_prob,
            "entropy": entropy,
            "length": length,
            "mask": mask,
            "logits": logits,
        }


def main(config: dict):
    cfg: Config = Config(**config)

    # model
    agents = {
        f"A{i}": Agent(
            input_encoder=DiscSeqMLPEncoder(
                length=cfg.input_length,
                n_values=cfg.input_n_values,
                output_dim=cfg.latent_dim,
                hidden_dim=cfg.mlp_encoder_hidden_dim,
                embed_dim=cfg.mlp_encoder_embed_dim,
                activation=cfg.mlp_encoder_activation,
                use_layer_norm=cfg.mlp_encoder_use_layer_norm,
                use_residual=cfg.mlp_encoder_use_residual,
                n_blocks=cfg.mlp_encoder_n_blocks,
            ),
            input_decoder=DiscSeqMLPDecoder(
                length=cfg.input_length,
                n_values=cfg.input_n_values,
                input_dim=cfg.latent_dim,
                hidden_dim=cfg.mlp_decoder_hidden_dim,
                activation=cfg.mlp_decoder_activation,
                use_layer_norm=cfg.mlp_decoder_use_layer_norm,
                use_residual=cfg.mlp_decoder_use_residual,
                n_blocks=cfg.mlp_decoder_n_blocks,
            ),
            message_encoder=DiscSeqRNNEncoder(
                n_values=cfg.message_n_values,
                output_dim=cfg.latent_dim,
                hidden_dim=cfg.rnn_encoder_hidden_dim,
                embed_dim=cfg.rnn_encoder_embed_dim,
                rnn_type=cfg.rnn_encoder_rnn_type,
                n_layers=cfg.rnn_encoder_n_layers,
            ),
            message_decoder=DiscSeqAdapter(
                DiscSeqRNNDecoder(
                    length=cfg.message_length,
                    n_values=cfg.message_n_values,
                    input_dim=cfg.latent_dim,
                    hidden_dim=cfg.rnn_decoder_hidden_dim,
                    embed_dim=cfg.rnn_decoder_embed_dim,
                    rnn_type=cfg.rnn_decoder_rnn_type,
                    n_layers=cfg.rnn_decoder_n_layers,
                ),
                padding=True,
            ),
        ).to(cfg.device)
        for i in range(cfg.n_agents)
    }
    train(
        agents,
        {
            "n_epochs": cfg.n_epochs,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
            "device": cfg.device,
            "exp_name": cfg.exp_name,
            "wandb_project": cfg.wandb_project,
            "use_tqdm": cfg.use_tqdm,
            "lr": cfg.lr,
            "input_length": cfg.input_length,
            "input_n_values": cfg.input_n_values,
            "message_length": cfg.message_length,
            "message_n_values": cfg.message_n_values,
            "entropy_weight": cfg.entropy_weight,
            "length_weight": cfg.length_weight,
            "baseline": cfg.baseline,
            "length_baseline": cfg.length_baseline,
        },
    )
