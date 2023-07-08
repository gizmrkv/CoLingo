import datetime
import json
import os
import uuid
from dataclasses import dataclass
from itertools import product

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ..core import Callback, Runner, fix_seed
from ..dataset import random_split
from ..game import (
    InferringGame,
    InferringGameEvaluator,
    InferringGameResult,
    InferringGameTrainer,
)
from ..logger import Logger, WandBLogger
from ..model import DiscSeqMLPDecoder, DiscSeqMLPEncoder


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
    length: int
    n_values: int
    latent_dim: int
    embed_dim: int

    # encoder config
    encoder_hidden_dim: int
    encoder_activation: str
    encoder_use_layer_norm: bool
    encoder_use_residual: bool
    encoder_n_blocks: int

    # decoder config
    decoder_hidden_dim: int
    decoder_activation: str
    decoder_use_layer_norm: bool
    decoder_use_residual: bool
    decoder_n_blocks: int


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


def run_disc_seq_mlp_exp(config: dict):
    cfg: Config = Config(**config)

    assert cfg.device in ["cpu", "cuda"], "Invalid device"
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Use CPU instead.")
        cfg.device = "cpu"

    now = datetime.datetime.now()
    log_id = str(uuid.uuid4())[-4:]
    log_dir = f"log/{cfg.exp_name}_{now.date()}_{now.strftime('%H%M%S')}_{log_id}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    fix_seed(cfg.seed)

    encoder = DiscSeqMLPEncoder(
        length=cfg.length,
        n_values=cfg.n_values,
        output_dim=cfg.latent_dim,
        hidden_dim=cfg.encoder_hidden_dim,
        embed_dim=cfg.embed_dim,
        activation=cfg.encoder_activation,
        use_layer_norm=cfg.encoder_use_layer_norm,
        use_residual=cfg.encoder_use_residual,
        n_blocks=cfg.encoder_n_blocks,
    )
    decoder = DiscSeqMLPDecoder(
        length=cfg.length,
        n_values=cfg.n_values,
        input_dim=cfg.latent_dim,
        hidden_dim=cfg.decoder_hidden_dim,
        activation=cfg.decoder_activation,
        use_layer_norm=cfg.decoder_use_layer_norm,
        use_residual=cfg.decoder_use_residual,
        n_blocks=cfg.decoder_n_blocks,
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

    class EarlyStopLogger(Logger):
        def __init__(self):
            self._stop = False

        def log(self, met: dict):
            if met["valid"]["A"]["acc_comp"] >= 0.9999:
                self._stop = True

        def __call__(self, iteration: int):
            return self._stop

    wandb_logger = WandBLogger(project=cfg.wandb_project)
    early_stopper = EarlyStopLogger()

    train_eval = InferringGameEvaluator(
        game=game,
        agents=agents,
        input=train_dataset,
        metric=metric,
        logger=wandb_logger,
        name="train",
    )
    valid_eval = InferringGameEvaluator(
        game=game,
        agents=agents,
        input=valid_dataset,
        metric=metric,
        logger=[wandb_logger, early_stopper],
        name="valid",
    )

    class CountSteps(Callback):
        def __init__(self, logger: Logger):
            self.logger = logger
            self._steps = 0

        def on_update(self, iteration: int):
            self._steps += 1

        def on_end(self):
            self.logger.log({"steps": self._steps})

    runner = Runner(
        trainer,
        train_eval,
        valid_eval,
        CountSteps(wandb_logger),
        wandb_logger,
        early_stop=early_stopper,
    )
    runner.run(cfg.n_epochs)
