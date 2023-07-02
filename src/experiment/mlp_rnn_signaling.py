import datetime
import json
import os
import uuid
from dataclasses import dataclass
from itertools import combinations, product
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from torchtyping import TensorType

from ..analysis import language_similarity, topographic_similarity
from ..baseline import BatchMeanBaseline
from ..core import Runner
from ..dataset import random_split
from ..game import (
    CollectiveInferringGame,
    CollectiveInferringGameEvaluator,
    CollectiveInferringGameResult,
    InferringGame,
    InferringGameEvaluator,
    InferringGameResult,
    SignalingGame,
    SignalingGameEvaluator,
    SignalingGameResult,
    SignalingGameTrainer,
)
from ..logger import WandBLogger
from ..loss import DiscSeqReinforceLoss
from ..model import (
    DiscSeqMLPDecoder,
    DiscSeqMLPEncoder,
    DiscSeqRNNDecoder,
    DiscSeqRNNEncoder,
)
from ..scheduler import IntervalScheduler
from ..util import ModelInitializer, fix_seed


@dataclass
class Config:
    # exp config
    n_epochs: int
    batch_size: int
    seed: int
    device: str
    exp_name: str
    wandb_project: str

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

        return x, (log_prob, entropy, length, mask)


def run_mlp_rnn_signaling_exp(config: dict):
    # make config
    cfg = Config(**config)

    # check device
    assert cfg.device in ["cpu", "cuda"]
    assert cfg.device == "cpu" or torch.cuda.is_available()

    # make log dir
    now = datetime.datetime.now()
    log_id = str(uuid.uuid4())[-4:]
    log_dir = f"log/{cfg.exp_name}_{now.date()}_{now.strftime('%H%M%S')}_{log_id}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save config
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    fix_seed(cfg.seed)

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

    optimziers = {
        name: optim.Adam(agent.parameters(), lr=cfg.lr)
        for name, agent in agents.items()
    }

    agent_initializer = ModelInitializer(
        model=agents.values(),
    )

    dataset = (
        torch.Tensor(
            list(product(torch.arange(cfg.input_n_values), repeat=cfg.input_length))
        )
        .long()
        .to(cfg.device)
    )
    train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(
        TensorDataset(train_dataset, train_dataset),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    baselines = {"batch_mean": BatchMeanBaseline}
    disc_seq_rf_loss = DiscSeqReinforceLoss(
        entropy_weight=cfg.entropy_weight,
        length_weight=cfg.length_weight,
        baseline=baselines[cfg.baseline](),
        length_baseline=baselines[cfg.length_baseline](),
    )

    def loss(result: SignalingGameResult, target: torch.Tensor):
        out_logits_r = result.output_info_r
        out_logits_r = out_logits_r.view(-1, cfg.input_n_values)
        loss_out_r = F.cross_entropy(out_logits_r, target.view(-1), reduction="none")
        loss_out_r = loss_out_r.view(-1, cfg.input_length).sum(dim=-1)

        log_prob_msg_s, entropy_msg_s, length_msg_s, _ = result.message_info_s
        loss_msg_s = disc_seq_rf_loss(
            reward=-loss_out_r.detach(),
            log_prob=log_prob_msg_s,
            entropy=entropy_msg_s,
            length=length_msg_s,
        )

        return loss_out_r + loss_msg_s

    game = SignalingGame()
    trainer = SignalingGameTrainer(
        game=game,
        agents=agents,
        optimizers=optimziers,
        dataloader=train_dataloader,
        loss=loss,
    )

    def metric(result: SignalingGameResult):
        mark = result.output_r == result.input
        acc_comp = mark.all(dim=-1).float().mean().item()
        acc = mark.float().mean(dim=0)
        acc_part = acc.mean().item()
        met = {"acc_comp": acc_comp, "acc_part": acc_part}
        met |= {f"acc{i}": a for i, a in enumerate(list(acc))}
        return met

    loggers = [WandBLogger(project=cfg.wandb_project)]

    evaluators = [
        SignalingGameEvaluator(
            game=game,
            agents=agents,
            input=input,
            metric=metric,
            logger=loggers,
            name=name,
        )
        for name, input in [("train", train_dataset), ("valid", valid_dataset)]
    ]

    def drop_padding(x: np.ndarray):
        i = np.argwhere(x == 0)
        return x if len(i) == 0 else x[: i[0, 0]]

    def lansim_metric(result: CollectiveInferringGameResult):
        names = list(result.agents)
        met = {}
        for name1, name2 in combinations(names, 2):
            output1: torch.Tensor = result.outputs[name1]
            output2: torch.Tensor = result.outputs[name2]
            output1 = output1.cpu().numpy()
            output2 = output2.cpu().numpy()

            lansim = language_similarity(
                output1,
                output2,
                dist="Levenshtein",
                processor=drop_padding,
                normalized=True,
            )
            met[f"{name1}-{name2}"] = lansim

        met["mean"] = np.mean(list(met.values()))
        return met

    def topsim_metric(result: CollectiveInferringGameResult):
        input = result.input.cpu().numpy()
        met = {}
        for name in result.agents:
            output = result.outputs[name].cpu().numpy()
            topsim = topographic_similarity(
                input, output, y_processor=drop_padding, workers=-1
            )
            met[name] = topsim

        met["mean"] = np.mean(list(met.values()))
        return met

    lansim_game = CollectiveInferringGame(output_command="send")
    lansim_evaluators = [
        CollectiveInferringGameEvaluator(
            game=lansim_game,
            agents=agents,
            input=input,
            metric=lansim_metric,
            logger=loggers,
            name=name,
        )
        for name, input in [
            ("train_sync", train_dataset),
            ("valid_sync", valid_dataset),
        ]
    ]
    topsim_evaluators = [
        CollectiveInferringGameEvaluator(
            game=lansim_game,
            agents=agents,
            input=input,
            metric=topsim_metric,
            logger=loggers,
            name=name,
        )
        for name, input in [
            ("train_topo", train_dataset),
            ("valid_topo", valid_dataset),
        ]
    ]

    callbacks = [
        trainer,
        *evaluators,
        *loggers,
        IntervalScheduler(lansim_evaluators, 100),
        IntervalScheduler(topsim_evaluators, 500),
        IntervalScheduler(agent_initializer, 100000),
    ]

    runner = Runner(callbacks)
    runner.run(cfg.n_epochs)
