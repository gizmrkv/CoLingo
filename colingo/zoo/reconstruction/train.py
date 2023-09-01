import datetime
import json
import os
import uuid
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, List, Mapping

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ...core import Evaluator, Runner, RunnerCallback, Trainer
from ...game import ReconstructionGame
from ...loggers import WandbLogger
from ...utils import (
    DuplicateChecker,
    MetricsEarlyStopper,
    StepCounter,
    Stopwatch,
    Timer,
    fix_seed,
    init_weights,
    random_split,
)
from .agent import Decoder, Encoder
from .loss import loss
from .metrics import Metrics


@dataclass
class Config:
    zoo: str

    n_epochs: int
    batch_size: int
    device: str
    wandb_project: str
    use_tqdm: bool

    lr: float
    length: int
    values: int

    metrics_interval: int


def train(
    encoder: Encoder,
    decoder: Decoder,
    config: Mapping[str, Any],
    additions: Iterable[RunnerCallback] | None = None,
) -> None:
    cfg = Config(**{k: config[k] for k in Config.__dataclass_fields__})

    models: List[nn.Module] = [encoder, decoder]
    optimizers = [optim.Adam(model.parameters(), lr=cfg.lr) for model in models]

    for model in [encoder, decoder]:
        model.to(cfg.device)
        model.apply(init_weights)

    dataset = (
        torch.Tensor(list(product(torch.arange(cfg.values), repeat=cfg.length)))
        .long()
        .to(cfg.device)
    )
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    game = ReconstructionGame(encoder, decoder)
    trainer = Trainer(
        agents=models,
        input=train_dataloader,
        games=[game],
        loss=loss,
        optimizers=optimizers,
    )

    wandb_logger = WandbLogger(project=cfg.wandb_project)
    duplicate_checker = DuplicateChecker()
    early_stopper = MetricsEarlyStopper(
        lambda metrics: metrics["test.acc_comp"] > 0.99
        if "test.acc_comp" in metrics
        else False
    )
    evaluators = []
    for name, input in [
        ("train", train_dataloader),
        ("test", test_dataloader),
    ]:
        evaluators.append(
            Evaluator(
                agents=models,
                input=input,
                games=[game],
                callbacks=[
                    Metrics(name, [wandb_logger, early_stopper, duplicate_checker])
                ],
                intervals=[cfg.metrics_interval],
            )
        )

    runner_callbacks = [
        *(additions or []),
        trainer,
        *evaluators,
        StepCounter("step", [wandb_logger, duplicate_checker]),
        Stopwatch("time", [wandb_logger, duplicate_checker]),
        wandb_logger,
        early_stopper,
        duplicate_checker,
    ]
    # runner_callbacks = [Timer(runner_callbacks)]
    runner = Runner(runner_callbacks, stopper=early_stopper, use_tqdm=cfg.use_tqdm)
    runner.run(cfg.n_epochs)
