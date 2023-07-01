from dataclasses import dataclass
from itertools import product

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ..core import Runner
from ..dataset import random_split
from ..game import (
    InferringGame,
    InferringGameEvaluator,
    InferringGameResult,
    InferringGameTrainer,
)
from ..logger import WandBLogger
from ..model import (
    DiscSeqMLPDecoder,
    DiscSeqMLPEncoder,
    DiscSeqRNNDecoder,
    DiscSeqRNNEncoder,
)


@dataclass
class Config:
    # exp config
    n_epochs: int
    batch_size: int
    seed: int
    device: str
    wandb_project: str

    # common config
    lr: float
    length: int
    n_values: int
    latent_dim: int

    # encoder config
    encoder_hidden_dim: int
    encoder_embed_dim: int
    encoder_rnn_type: str
    encoder_n_layers: int

    # decoder config
    decoder_hidden_dim: int
    decoder_embed_dim: int
    decoder_rnn_type: str
    decoder_n_layers: int


class Agent(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        input: torch.Tensor | None = None,
        latent: torch.Tensor | None = None,
        command: str | None = None,
    ):
        match command:
            case "input":
                return self.encoder(input)
            case "output":
                return self.decoder(latent)
            case _:
                raise ValueError(f"Unknown command: {command}")


def run_disc_seq_rnn_exp(cfg: dict):
    cfg: Config = Config(**cfg)

    encoder = DiscSeqRNNEncoder(
        n_values=cfg.n_values,
        output_dim=cfg.latent_dim,
        hidden_dim=cfg.encoder_hidden_dim,
        embed_dim=cfg.encoder_embed_dim,
        rnn_type=cfg.encoder_rnn_type,
        n_layers=cfg.encoder_n_layers,
    )
    decoder = DiscSeqRNNDecoder(
        length=cfg.length,
        n_values=cfg.n_values,
        input_dim=cfg.latent_dim,
        hidden_dim=cfg.decoder_hidden_dim,
        embed_dim=cfg.decoder_embed_dim,
        rnn_type=cfg.decoder_rnn_type,
        n_layers=cfg.decoder_n_layers,
    )
    agent = Agent(encoder, decoder).to(cfg.device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr)

    agents = {"A": agent}
    optimizers = {"A": optimizer}

    dataset = (
        torch.Tensor(list(product(torch.arange(cfg.n_values), repeat=cfg.length)))
        .long()
        .to(cfg.device)
    )
    train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(
        TensorDataset(train_dataset, train_dataset),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    def loss(result: InferringGameResult, target: torch.Tensor):
        logits = result.info.view(-1, cfg.n_values)
        target = target.view(-1)
        return F.cross_entropy(logits, target)

    game = InferringGame()
    trainer = InferringGameTrainer(
        game=game,
        agents=agents,
        optimizers=optimizers,
        dataloader=train_dataloader,
        loss=loss,
    )

    def metric(result: InferringGameResult):
        mark = result.output == result.input
        acc_comp = mark.all(dim=-1).float().mean().item()
        acc = mark.float().mean(dim=0)
        acc_part = acc.mean().item()
        met = {"acc_comp": acc_comp, "acc_part": acc_part}
        met |= {f"acc{i}": a for i, a in enumerate(list(acc))}
        return met

    loggers = [WandBLogger(project=cfg.wandb_project)]

    evaluators = [
        InferringGameEvaluator(
            game=game,
            agents=agents,
            input=input,
            metric=metric,
            logger=loggers,
            name=name,
        )
        for name, input in [("train", train_dataset), ("valid", valid_dataset)]
    ]

    callbacks = [
        trainer,
        *evaluators,
        *loggers,
    ]

    runner = Runner(callbacks)
    runner.run(cfg.n_epochs)
