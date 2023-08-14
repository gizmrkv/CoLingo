import datetime
import json
import os
import uuid
from dataclasses import dataclass
from itertools import product
from typing import Any, List, Mapping

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ...core import Evaluator, Runner, Trainer
from ...game import ReconstructionGame
from ...loggers import WandbLogger
from ...utils import (
    DuplicateChecker,
    Interval,
    MetricsEarlyStopper,
    StepCounter,
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
    n_values: int

    metrics_interval: int

    seed: int | None = None


def train(encoder: Encoder, decoder: Decoder, config: Mapping[str, Any]) -> None:
    cfg = Config(**{k: config[k] for k in Config.__dataclass_fields__})

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Use CPU instead.")
        cfg.device = "cpu"

    now = datetime.datetime.now()
    run_id = str(uuid.uuid4())[-4:]
    run_name = f"{now.date()}_{now.strftime('%H%M%S')}_{run_id}"
    log_dir = f"log/{run_name}_{cfg.zoo}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    if cfg.seed is not None:
        fix_seed(cfg.seed)

    models: List[nn.Module] = [encoder, decoder]
    optimizers = [optim.Adam(model.parameters(), lr=cfg.lr) for model in models]

    for model in [encoder, decoder]:
        model.to(cfg.device)
        model.apply(init_weights)

    dataset = (
        torch.Tensor(list(product(torch.arange(cfg.n_values), repeat=cfg.length)))
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

    wandb_logger = WandbLogger(project=cfg.wandb_project, name=run_name)
    duplicate_checker = DuplicateChecker()
    early_stopper = MetricsEarlyStopper(
        lambda metrics: metrics["test.acc_comp"] > 0.99
        if "test.acc_comp" in metrics
        else False
    )
    train_metrics = Metrics(
        "train", cfg.length, cfg.n_values, [wandb_logger, duplicate_checker]
    )
    test_metrics = Metrics(
        "test",
        cfg.length,
        cfg.n_values,
        [wandb_logger, early_stopper, duplicate_checker],
    )
    train_evaluator = Evaluator(
        agents=models,
        input=train_dataloader,
        games=[game],
        callbacks=[train_metrics],
    )
    test_evaluator = Evaluator(
        agents=models,
        input=test_dataloader,
        games=[game],
        callbacks=[test_metrics],
    )

    runner_callbacks = [
        trainer,
        Interval(cfg.metrics_interval, [train_evaluator, test_evaluator]),
        StepCounter("step", [wandb_logger, duplicate_checker]),
        wandb_logger,
        early_stopper,
        duplicate_checker,
    ]
    # runner_callbacks = [Timer(runner_callbacks)]
    runner = Runner(runner_callbacks, stopper=early_stopper)
    runner.run(cfg.n_epochs)
