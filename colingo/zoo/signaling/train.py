import datetime
import json
import os
import uuid
from dataclasses import asdict, dataclass
from itertools import product
from typing import Literal

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from ...baseline import BatchMeanBaseline
from ...core import Runner
from ...logger import DuplicateChecker, HeatmapLogger, WandBLogger
from ...loss import ReinforceLoss
from ...utils import (
    StepCounter,
    TimeWatcher,
    Trainer,
    fix_seed,
    init_weights,
    interval,
    random_split,
    shuffle,
)
from .agent import Agent
from .game import Game
from .loss import (
    Loss,
    ReceiverAutoEncodingCrossEntropyLoss,
    ReceiverMessageCrossEntropyLoss,
    ReceiverObjectCrossEntropyLoss,
    SenderAutoEncodingCrossEntropyLoss,
    SenderAutoEncodingLeaveCrossEntropyLoss,
    SenderMessageReinforceLoss,
    SenderObjectCrossEntropyLoss,
)
from .metrics import GameMetrics, LanguageSaver, LanguageSimilarityMetrics


@dataclass
class Config:
    # exp config
    n_epochs: int
    batch_size: int
    device: str
    zoo_name: str
    wandb_project: str
    use_tqdm: bool

    # common config
    lr: float
    object_length: int
    object_n_values: int
    message_length: int
    message_n_values: int

    # optional config
    seed: int | None = None

    run_sender_output: bool = False
    run_receiver_send: bool = False
    run_sender_auto_encoding: bool = False
    run_receiver_auto_encoding: bool = False

    receiver_loss_weight: float = 1.0

    sender_loss_weight: float = 1.0
    baseline: str = "batch_mean"
    length_baseline: str = "batch_mean"
    entropy_weight: float = 0.0
    length_weight: float = 0.0

    raece_loss_weight: float | None = None
    rmce_loss_weight: float | None = None
    saece_loss_weight: float | None = None
    soce_loss_weight: float | None = None
    saecel_loss_weight: float | None = None

    eval_interval: int = 10
    acc_heatmap_interval: int = 5
    topsim_interval: int = 5
    lansim_interval: int = 50
    lansim_heatmap_interval: int = 1
    language_save_interval: int = 50


def train(
    agents: dict[str, Agent],
    adjacency: dict[str, list[str]],
    cfg: Config,
) -> None:
    # pre process
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Use CPU instead.")
        cfg.device = "cpu"

    now = datetime.datetime.now()
    log_id = str(uuid.uuid4())[-4:]
    log_dir = f"log/{cfg.zoo_name}_{now.date()}_{now.strftime('%H%M%S')}_{log_id}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=4)

    if cfg.seed is not None:
        fix_seed(cfg.seed)

    for agent in agents.values():
        agent.to(cfg.device)
        agent.apply(init_weights)

    optimizers = {
        name: optim.Adam(agent.parameters(), lr=cfg.lr)
        for name, agent in agents.items()
    }

    # data
    dataset = (
        torch.Tensor(
            list(product(torch.arange(cfg.object_n_values), repeat=cfg.object_length))
        )
        .long()
        .to(cfg.device)
    )
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    dataloader = DataLoader(dataset, batch_size=dataset.shape[0])  # type: ignore
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True  # type: ignore
    )
    test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.shape[0])  # type: ignore

    # trainer
    train_losses: list[nn.Module] = []
    eval_losses: dict[str, nn.Module] = {}
    if cfg.raece_loss_weight is not None:
        train_losses.append(
            ReceiverAutoEncodingCrossEntropyLoss(
                cfg.message_length, cfg.message_n_values, cfg.raece_loss_weight
            )
        )
        eval_losses["raece"] = ReceiverAutoEncodingCrossEntropyLoss(
            cfg.message_length, cfg.message_n_values
        )
    if cfg.rmce_loss_weight is not None:
        train_losses.append(
            ReceiverMessageCrossEntropyLoss(
                cfg.message_length, cfg.message_n_values, cfg.rmce_loss_weight
            )
        )
        eval_losses["rmce"] = ReceiverMessageCrossEntropyLoss(
            cfg.message_length, cfg.message_n_values
        )
    if cfg.saece_loss_weight is not None:
        train_losses.append(
            SenderAutoEncodingCrossEntropyLoss(
                cfg.object_length, cfg.object_n_values, cfg.saece_loss_weight
            )
        )
        eval_losses["saece"] = SenderAutoEncodingCrossEntropyLoss(
            cfg.object_length, cfg.object_n_values
        )
    if cfg.soce_loss_weight is not None:
        train_losses.append(
            SenderObjectCrossEntropyLoss(
                cfg.object_length, cfg.object_n_values, cfg.soce_loss_weight
            )
        )
        eval_losses["soce"] = SenderObjectCrossEntropyLoss(
            cfg.object_length, cfg.object_n_values
        )
    if cfg.saecel_loss_weight is not None:
        train_losses.append(
            SenderAutoEncodingLeaveCrossEntropyLoss(
                cfg.message_length, cfg.message_n_values, cfg.saecel_loss_weight
            )
        )
        eval_losses["saecel"] = SenderAutoEncodingLeaveCrossEntropyLoss(
            cfg.message_length, cfg.message_n_values
        )

    baselines = {"batch_mean": BatchMeanBaseline}
    reinforce_loss = ReinforceLoss(
        cfg.entropy_weight,
        cfg.length_weight,
        baselines[cfg.baseline](),
        baselines[cfg.length_baseline](),
    )
    sender_loss = SenderMessageReinforceLoss(reinforce_loss, cfg.sender_loss_weight)
    receiver_loss = ReceiverObjectCrossEntropyLoss(
        cfg.object_length, cfg.object_n_values, cfg.receiver_loss_weight
    )
    loss = Loss(
        cfg.object_length,
        cfg.object_n_values,
        cfg.message_length,
        cfg.message_n_values,
        sender_loss,
        receiver_loss,
        train_losses,
    )
    eval_losses["roce"] = receiver_loss
    eval_losses["signaling"] = loss

    games = []
    for name_s, names_r in adjacency.items():
        sender = agents[name_s]
        receivers = [agents[name_r] for name_r in names_r]
        games.append(
            Game(
                sender,
                receivers,
                cfg.run_sender_output,
                cfg.run_receiver_send,
                cfg.run_sender_auto_encoding,
                cfg.run_receiver_auto_encoding,
            )
        )

    trainer = Trainer(games, optimizers.values(), train_dataloader, loss)

    # evaluator
    loggers = [WandBLogger(cfg.wandb_project, name=log_dir), DuplicateChecker()]
    heatmap_option = {
        "vmin": 0,
        "vmax": 1,
        "cmap": "viridis",
        "annot": True,
        "fmt": ".2f",
        "cbar": True,
        "square": True,
    }
    heatmap_loggers = []
    game_metrics = []
    lansim_metrics = []
    for name, data in [("train", train_dataloader), ("test", test_dataloader)]:
        acc_comp_heatmap_logger = HeatmapLogger(
            log_dir,
            f"{name}.acc_comp",
            loggers,
            heatmap_option=heatmap_option,
        )
        acc_part_heatmap_logger = HeatmapLogger(
            log_dir,
            f"{name}.acc_part",
            loggers,
            heatmap_option=heatmap_option,
        )
        game_metrics.append(
            GameMetrics(
                name,
                agents,
                data,
                eval_losses,
                cfg.topsim_interval,
                cfg.acc_heatmap_interval,
                loggers,
                [acc_comp_heatmap_logger],
                [acc_part_heatmap_logger],
                {
                    "run_sender_output": cfg.run_sender_output,
                    "run_receiver_send": cfg.run_receiver_send,
                    "run_sender_auto_encoding": cfg.run_sender_auto_encoding,
                    "run_receiver_auto_encoding": cfg.run_receiver_auto_encoding,
                },
            )
        )

        lansim_heatmap_logger = HeatmapLogger(
            log_dir,
            f"{name}.lansim",
            loggers,
            heatmap_option=heatmap_option,
        )
        lansim_metrics.append(
            LanguageSimilarityMetrics(
                name,
                agents,
                data,
                cfg.lansim_heatmap_interval,
                loggers,
                [lansim_heatmap_logger],
            )
        )

        heatmap_loggers.append(acc_comp_heatmap_logger)
        heatmap_loggers.append(acc_part_heatmap_logger)
        heatmap_loggers.append(lansim_heatmap_logger)

    # runner
    callbacks = [
        trainer,
        *heatmap_loggers,
        interval(cfg.eval_interval, game_metrics),
        interval(cfg.lansim_interval, lansim_metrics),
        StepCounter("total_steps", loggers),
        interval(
            cfg.language_save_interval, [LanguageSaver(log_dir, agents, dataloader)]
        ),
        *loggers,
    ]
    # callbacks = [TimeWatcher(callbacks)]
    runner = Runner(callbacks, use_tqdm=cfg.use_tqdm)
    runner.run(cfg.n_epochs)
