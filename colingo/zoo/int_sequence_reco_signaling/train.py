import datetime
import json
import os
import uuid
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Callable, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.typing import NDArray
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torchtyping import TensorType

from ...analysis import topographic_similarity
from ...core import Evaluator, Runner, Trainer
from ...game import IDecoder, IEncoder, ReconstructionGame, ReconstructionGameResult
from ...loggers import WandbLogger
from ...loss import ReinforceLoss
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
    n_epochs: int
    batch_size: int
    device: str
    seed: int
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


def train(encoder: Encoder, decoder: Decoder, config: dict[str, Any]) -> None:
    cfg = Config(**{k: config[k] for k in Config.__dataclass_fields__})

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Use CPU instead.")
        cfg.device = "cpu"

    now = datetime.datetime.now()
    run_id = str(uuid.uuid4())[-4:]
    run_name = f"{now.date()}_{now.strftime('%H%M%S')}_{run_id}"
    log_dir = f"log/int_sequence_auto_encoding_{run_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=4)

    fix_seed(cfg.seed)

    models: list[nn.Module] = [encoder, decoder]
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
        train_dataset, batch_size=cfg.batch_size, shuffle=True  # type: ignore
    )
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))  # type: ignore

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
    train_metrics = Metrics(
        name="train",
        object_length=cfg.object_length,
        object_n_values=cfg.object_n_values,
        message_length=cfg.message_length,
        message_n_values=cfg.message_n_values,
        loss=loss,
        callbacks=[wandb_logger, duplicate_checker],
    )
    test_metrics = Metrics(
        name="test",
        object_length=cfg.object_length,
        object_n_values=cfg.object_n_values,
        message_length=cfg.message_length,
        message_n_values=cfg.message_n_values,
        loss=loss,
        callbacks=[wandb_logger, early_stopper, duplicate_checker],
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

    train_topsim = TopographicSimilarityMetrics(
        "train", [wandb_logger, duplicate_checker]
    )
    test_topsim = TopographicSimilarityMetrics(
        "test", [wandb_logger, duplicate_checker]
    )
    train_topsim_evaluator = Evaluator(
        agents=models,
        input=train_dataloader,
        games=[game],
        callbacks=[train_topsim],
    )
    test_topsim_evaluator = Evaluator(
        agents=models,
        input=test_dataloader,
        games=[game],
        callbacks=[test_topsim],
    )

    language_logger = LanguageLogger(log_dir, "lang")
    language_logger_evaluator = Evaluator(
        agents=models,
        input=DataLoader(dataset, batch_size=len(dataset)),  # type: ignore
        games=[game],
        callbacks=[language_logger],
    )

    runner_callbacks = [
        trainer,
        Interval(cfg.metrics_interval, [train_evaluator, test_evaluator]),
        Interval(cfg.topsim_interval, [train_topsim_evaluator, test_topsim_evaluator]),
        Interval(cfg.language_log_interval, [language_logger_evaluator]),
        StepCounter("step", [wandb_logger, duplicate_checker]),
        wandb_logger,
        early_stopper,
        duplicate_checker,
    ]
    # runner_callbacks = [Timer(runner_callbacks)]
    runner = Runner(runner_callbacks, stopper=early_stopper)
    runner.run(cfg.n_epochs)
