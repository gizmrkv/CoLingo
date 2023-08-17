import datetime
import json
import os
import uuid
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Mapping, Set

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtyping import TensorType

import wandb

from ...core import Evaluator, Runner, RunnerCallback, Trainer
from ...game import ReconstructionNetworkGame
from ...loggers import HeatmapLogger, WandbLogger
from ...utils import (
    DuplicateChecker,
    EarlyStopper,
    StepCounter,
    Timer,
    fix_seed,
    init_weights,
    random_split,
)
from .agent import Agent, MessageAuxiliary
from .loss import Loss
from .metrics import (
    AccuracyHeatmapLogger,
    LanguageLogger,
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
    object_n_values: int
    message_length: int
    message_n_values: int

    entropy_weight: float
    length_weight: float

    metrics_interval: int
    topsim_interval: int
    language_log_interval: int
    acc_heatmap_interval: int
    lansim_interval: int

    decoder_ae: bool

    seed: int | None = None


def train(
    agents: Mapping[str, Agent],
    game: ReconstructionNetworkGame[
        TensorType[..., int],
        TensorType[..., int],
        MessageAuxiliary,
        TensorType[..., float],
    ],
    config: Mapping[str, Any],
    game_editor: RunnerCallback | None = None,
) -> None:
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

    optimizers = {
        name: optim.Adam(agent.parameters(), lr=cfg.lr)
        for name, agent in agents.items()
    }

    for agent in agents.values():
        agent.to(cfg.device)
        agent.apply(init_weights)

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

    def baseline(x: TensorType[..., float]) -> TensorType[..., float]:
        return x.detach().mean(dim=0)

    loss = Loss(
        agents=agents,
        object_length=cfg.object_length,
        object_n_values=cfg.object_n_values,
        message_length=cfg.message_length,
        message_n_values=cfg.message_n_values,
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

    wandb_logger = WandbLogger(project=cfg.wandb_project, name=run_name)
    duplicate_checker = DuplicateChecker()

    adj_comp: Dict[str, Set[str]] = {s: {t for t in agents} for s in agents}
    game_comp: ReconstructionNetworkGame[
        TensorType[..., int],
        TensorType[..., int],
        MessageAuxiliary,
        TensorType[..., float],
    ] = ReconstructionNetworkGame(agents, adj_comp)

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

        def __call__(self, path: str) -> None:
            wandb_logger({f"{self.name}": wandb.Video(path)})

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
            save_dir=os.path.join(log_dir, f"{name}_acc_comp"),
            heatmap_option=heatmap_option,
            callbacks=[
                WandbHeatmapLogger(f"{name}.acc_comp"),
            ],
        )
        acc_part_heatmap = HeatmapLogger(
            save_dir=os.path.join(log_dir, f"{name}_acc_part"),
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
            save_dir=os.path.join(log_dir, f"{name}_lansim"),
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
    game_none: ReconstructionNetworkGame[
        TensorType[..., int],
        TensorType[..., int],
        MessageAuxiliary,
        TensorType[..., float],
    ] = ReconstructionNetworkGame(agents, adj_none)

    language_logger = LanguageLogger(os.path.join(log_dir, "lang"), agents)
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
        trainer,
        *evaluators,
        StepCounter("step", [wandb_logger, duplicate_checker]),
        *heatmap_loggers,
        wandb_logger,
        duplicate_checker,
    ]
    if game_editor is not None:
        runner_callbacks = [game_editor] + runner_callbacks
    # runner_callbacks = [Timer(runner_callbacks)]
    runner = Runner(runner_callbacks)
    runner.run(cfg.n_epochs)
