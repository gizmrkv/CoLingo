import datetime
import json
import os
import uuid
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Set

import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchtyping import TensorType

from ...core import Evaluator, RunnerCallback, TaskRunner, Trainer
from ...game import ReconstructionNetworkGame
from ...loggers import HeatmapLogger, WandbLogger
from ...utils import (
    EarlyStopper,
    KeyChecker,
    StepCounter,
    Stopwatch,
    Timer,
    fix_seed,
    init_weights,
    random_split,
)
from .agent import Agent, MessageAuxiliary
from .loss import Loss
from .metrics import (
    AccuracyHeatmapLogger,
    LangLogger,
    LanguageSimilarityMetrics,
    Metrics,
    TopographicSimilarityMetrics,
)


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
    object_values: int
    message_max_len: int
    message_vocab_size: int

    entropy_weight: float
    length_weight: float

    metrics_interval: int
    topsim_interval: int
    language_log_interval: int
    acc_heatmap_interval: int
    lansim_interval: int

    decoder_ae: bool


def train(
    agents: Mapping[str, Agent],
    game: ReconstructionNetworkGame[
        TensorType[..., int],
        TensorType[..., float],
        TensorType[..., int],
        None,
        TensorType[..., float],
        None,
        MessageAuxiliary,
    ],
    config: Mapping[str, Any],
    log_dir: Path,
    additions: Iterable[RunnerCallback] | None = None,
) -> None:
    cfg = Config(**{k: config[k] for k in Config.__dataclass_fields__})

    optimizers = {
        name: optim.Adam(agent.parameters(), lr=cfg.lr)
        for name, agent in agents.items()
    }

    for agent in agents.values():
        agent.to(cfg.device)
        agent.apply(init_weights)

    dataset = (
        torch.Tensor(
            list(product(torch.arange(cfg.object_values), repeat=cfg.object_length))
        )
        .long()
        .to(cfg.device)
    )
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

    def baseline(x: TensorType[..., float]) -> TensorType[..., float]:
        return x.detach().mean(dim=0)

    loss = Loss(
        agents=agents,
        object_length=cfg.object_length,
        object_values=cfg.object_values,
        message_max_len=cfg.message_max_len,
        message_vocab_size=cfg.message_vocab_size,
        entropy_weight=cfg.entropy_weight,
        length_weight=cfg.length_weight,
        baseline=baseline,
        length_baseline=baseline,
        decoder_ae=cfg.decoder_ae,
    )

    trainer = Trainer(
        agents=agents.values(),
        input=train_dataloader,
        games=[game],
        loss=loss,
        optimizers=optimizers.values(),
    )

    wandb_logger = WandbLogger(project=cfg.wandb_project)
    duplicate_checker = KeyChecker()

    adj_comp: Dict[str, Set[str]] = {s: {t for t in agents} for s in agents}
    game_comp = ReconstructionNetworkGame(agents, adj_comp)

    heatmap_option = {
        "vmin": 0,
        "vmax": 1,
        "cmap": "viridis",
        "annot": True,
        "fmt": ".2f",
        "cbar": True,
        "square": True,
    }

    class WandbHeatmapLogger:
        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, path: Path) -> None:
            wandb_logger({f"{self.name}": wandb.Video(path.as_posix())})

    evaluators = []
    heatmap_loggers = []
    for name, input in [
        ("train", train_dataloader),
        ("test", test_dataloader),
    ]:
        metrics = Metrics(
            name=name, loss=loss, callbacks=[wandb_logger, duplicate_checker]
        )
        topsim = TopographicSimilarityMetrics(
            name=name, callbacks=[wandb_logger, duplicate_checker]
        )

        acc_comp_heatmap = HeatmapLogger(
            save_dir=log_dir.joinpath(f"{name}_acc_comp"),
            heatmap_option=heatmap_option,
            callbacks=[
                WandbHeatmapLogger(f"{name}.acc_comp"),
            ],
        )
        acc_part_heatmap = HeatmapLogger(
            save_dir=log_dir.joinpath(f"{name}_acc_part"),
            heatmap_option=heatmap_option,
            callbacks=[
                WandbHeatmapLogger(f"{name}.acc_part"),
            ],
        )
        acc_heatmap = AccuracyHeatmapLogger(
            acc_comp_logger=acc_comp_heatmap,
            acc_part_logger=acc_part_heatmap,
        )
        heatmap_loggers.extend([acc_comp_heatmap, acc_part_heatmap])

        lansim_heatmap = HeatmapLogger(
            save_dir=log_dir.joinpath(f"{name}_lansim"),
            heatmap_option=heatmap_option,
            callbacks=[
                WandbHeatmapLogger(f"{name}.lansim"),
            ],
        )
        lansim = LanguageSimilarityMetrics(
            name=name,
            callbacks=[wandb_logger, duplicate_checker],
            heatmap_logger=lansim_heatmap,
        )
        heatmap_loggers.append(lansim_heatmap)

        evaluators.append(
            Evaluator(
                agents=agents.values(),
                input=input,
                games=[game_comp],
                callbacks=[metrics, topsim, acc_heatmap, lansim],
                intervals=[
                    cfg.metrics_interval,
                    cfg.topsim_interval,
                    cfg.acc_heatmap_interval,
                    cfg.lansim_interval,
                ],
            )
        )

    adj_none: Dict[str, Set[str]] = {s: set() for s in agents}
    game_none = ReconstructionNetworkGame(agents, adj_none)

    language_logger = LangLogger(log_dir.joinpath("lang"), agents)
    evaluators.append(
        Evaluator(
            agents=agents.values(),
            input=DataLoader(dataset, batch_size=len(dataset)),  # type: ignore
            games=[game_none],
            callbacks=[language_logger],
            intervals=[cfg.language_log_interval],
        )
    )

    runner_callbacks = [
        *(additions or []),
        trainer,
        *evaluators,
        StepCounter("step", [wandb_logger, duplicate_checker]),
        Stopwatch("time", [wandb_logger, duplicate_checker]),
        *heatmap_loggers,
        wandb_logger,
        duplicate_checker,
    ]
    # runner_callbacks = [Timer(runner_callbacks)]
    runner = TaskRunner(runner_callbacks, use_tqdm=cfg.use_tqdm)
    runner.run(cfg.n_epochs)
