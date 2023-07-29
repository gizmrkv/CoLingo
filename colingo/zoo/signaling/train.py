import datetime
import json
import os
import uuid
from dataclasses import asdict, dataclass
from itertools import combinations, product
from typing import Iterable, Literal

import torch
from torch import optim
from torch.utils.data import DataLoader

from ...core import Runner
from ...logger import DuplicateChecker, EarlyStopper, WandBLogger
from ...utils import (
    Evaluator,
    StepCounter,
    Trainer,
    fix_seed,
    init_weights,
    interval,
    random_split,
    shuffle,
)
from .agent import Agent
from .game import Game
from .loss import Loss
from .metrics import GameMetricsLogger, LanguageSimilarityMetrics


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

    use_reinforce: bool = False
    baseline: Literal["batch_mean"] = "batch_mean"
    entropy_weight: float = 0.0

    ruminate_weight: float = 0.0
    synchronize_weight: float = 0.0


def train(
    agents: dict[str, Agent], adjacency: dict[str, list[str]], cfg: Config
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
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True  # type: ignore
    )
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size)  # type: ignore

    # trainer
    loss = Loss(
        cfg.object_length,
        cfg.object_n_values,
        cfg.message_length,
        cfg.message_n_values,
        cfg.use_reinforce,
        cfg.baseline,
        cfg.entropy_weight,
        cfg.ruminate_weight,
        cfg.synchronize_weight,
    )

    trainers = []
    for name_s, names_r in adjacency.items():
        sender = agents[name_s]
        receivers = [agents[name_r] for name_r in names_r]
        game = Game(sender, receivers)
        trainer = Trainer(
            game,
            [optimizers[name_s], *[optimizers[name_r] for name_r in names_r]],
            train_dataloader,
            loss,
        )
        trainers.append(trainer)

    # evaluator
    wandb_logger = WandBLogger(cfg.wandb_project, name=log_dir)
    duplicate_checker = DuplicateChecker()
    names = list(agents.keys())
    train_evaluators = []
    test_evaluators = []
    metrics = []
    for name_s in names:
        sender = agents[name_s]
        receivers = [agents[name] for name in names]
        game = Game(sender, receivers)
        train_metrics = GameMetricsLogger(
            "train", name_s, 50, [wandb_logger, duplicate_checker]
        )
        test_metrics = GameMetricsLogger(
            "test", name_s, 50, [wandb_logger, duplicate_checker]
        )
        train_evaluators.append(Evaluator(game, train_dataloader, [train_metrics]))
        test_evaluators.append(Evaluator(game, test_dataloader, [test_metrics]))
        metrics.append(train_metrics)
        metrics.append(test_metrics)

    # train_acc_mat = AccuracyMatrix(
    #     cfg.object_length,
    #     "train",
    #     agents,
    #     train_dataloader,
    #     [wandb_logger, duplicate_checker],
    # )
    # test_acc_mat = AccuracyMatrix(
    #     cfg.object_length,
    #     "test",
    #     agents,
    #     test_dataloader,
    #     [wandb_logger, duplicate_checker],
    # )

    train_lansim_mat = LanguageSimilarityMetrics(
        log_dir,
        "train",
        agents,
        train_dataloader,
        50,
        [wandb_logger, duplicate_checker],
    )
    test_lansim_mat = LanguageSimilarityMetrics(
        log_dir, "test", agents, test_dataloader, 50, [wandb_logger, duplicate_checker]
    )

    # runner
    runner = Runner(
        [
            shuffle(trainers),
            interval(10, train_evaluators),
            interval(10, test_evaluators),
            StepCounter("total_steps", [wandb_logger, duplicate_checker]),
            *metrics,
            interval(10, [train_lansim_mat, test_lansim_mat]),
            wandb_logger,
            duplicate_checker,
        ],
        use_tqdm=cfg.use_tqdm,
    )
    runner.run(cfg.n_epochs)
