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
from torchtyping import TensorType

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
from .loss import Loss
from .metrics import LanguageLogger, Metrics, TopographicSimilarityMetrics


@dataclass
class Config:
    zoo: str

    n_epochs: int
    batch_size: int
    device: str
    wandb_project: str
    use_tqdm: bool

    lr: float
    object_length: int
    object_n_values: int
    message_length: int
    message_n_values: int

    entropy_weight: float
    length_weight: float

    metrics_interval: int
    topsim_interval: int
    language_log_interval: int

    seed: int | None = None


def train(encoder: Encoder, decoder: Decoder, config: Mapping[str, Any]) -> None:
    cfg = Config(**{k: config[k] for k in Config.__dataclass_fields__})

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
        torch.Tensor(
            list(product(torch.arange(cfg.object_n_values), repeat=cfg.object_length))
        )
        .long()
        .to(cfg.device)
    )
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

    game = ReconstructionGame(encoder, decoder)

    def baseline(x: TensorType[..., float]) -> TensorType[..., float]:
        return x.detach().mean(dim=0)

    loss = Loss(
        object_length=cfg.object_length,
        object_n_values=cfg.object_n_values,
        message_length=cfg.message_length,
        message_n_values=cfg.message_n_values,
        entropy_weight=cfg.entropy_weight,
        length_weight=cfg.length_weight,
        baseline=baseline,
        length_baseline=baseline,
    )

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
    metrics_evals = []
    topsim_evals = []
    for name, input in [
        ("train", train_dataloader),
        ("test", test_dataloader),
    ]:
        metrics = Metrics(
            name=name,
            loss=loss,
            callbacks=[wandb_logger, early_stopper, duplicate_checker],
        )
        metrics_evals.append(
            Evaluator(
                agents=models,
                input=input,
                games=[game],
                callbacks=[metrics],
            )
        )

        topsim = TopographicSimilarityMetrics(
            name=name, callbacks=[wandb_logger, duplicate_checker]
        )
        topsim_evals.append(
            Evaluator(
                agents=models,
                input=input,
                games=[game],
                callbacks=[topsim],
            )
        )

    language_logger = LanguageLogger(os.path.join(log_dir, "lang"))
    language_logger_evaluator = Evaluator(
        agents=models,
        input=DataLoader(dataset, batch_size=len(dataset)),  # type: ignore
        games=[game],
        callbacks=[language_logger],
    )

    runner_callbacks = [
        trainer,
        Interval(cfg.metrics_interval, metrics_evals),
        Interval(cfg.topsim_interval, topsim_evals),
        Interval(cfg.language_log_interval, [language_logger_evaluator]),
        StepCounter("step", [wandb_logger, duplicate_checker]),
        wandb_logger,
        early_stopper,
        duplicate_checker,
    ]
    # runner_callbacks = [Timer(runner_callbacks)]
    runner = Runner(runner_callbacks, stopper=early_stopper)
    runner.run(cfg.n_epochs)
