import datetime
import json
import os
import uuid
from dataclasses import asdict, dataclass
from itertools import product
from typing import Literal

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
    random_split,
)
from .agent import Agent
from .game import Game
from .loss import Loss
from .metrics import Metrics


@dataclass
class Config:
    # exp config
    n_epochs: int
    batch_size: int
    seed: int
    device: str
    zoo_name: str
    wandb_project: str
    use_tqdm: bool

    # common config
    lr: float
    length: int
    n_values: int

    # optional config
    use_reinforce: bool = False
    baseline: Literal["batch_mean"] = "batch_mean"
    entropy_weight: float = 0.0


def train(agent: Agent, cfg: Config) -> None:
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

    fix_seed(cfg.seed)

    agent.to(cfg.device)
    agent.apply(init_weights)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr)

    # data
    dataset = (
        torch.Tensor(list(product(torch.arange(cfg.n_values), repeat=cfg.length)))
        .long()
        .to(cfg.device)
    )
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True  # type: ignore
    )
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size)  # type: ignore

    # logger
    wandb_logger = WandBLogger(project=cfg.wandb_project, name=log_dir)
    early_stopper = EarlyStopper(lambda metrics: metrics["test.acc_comp"] > 0.99)
    duplicate_checker = DuplicateChecker()
    train_metrics = Metrics("train", [wandb_logger, duplicate_checker])
    test_metrics = Metrics("test", [wandb_logger, early_stopper, duplicate_checker])

    # game
    game = Game(agent)
    loss = Loss(cfg.n_values, cfg.use_reinforce, cfg.baseline, cfg.entropy_weight)
    trainer = Trainer(game, [optimizer], train_dataloader, loss, [train_metrics])
    test_evaluator = Evaluator(game, test_dataloader, [test_metrics])

    # run
    runner = Runner(
        [
            trainer,
            test_evaluator,
            StepCounter([wandb_logger]),
            wandb_logger,
            duplicate_checker,
        ],
        early_stop=early_stopper,
        use_tqdm=cfg.use_tqdm,
    )
    runner.run(cfg.n_epochs)
