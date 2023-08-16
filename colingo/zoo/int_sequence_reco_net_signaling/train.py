import datetime
import json
import os
import uuid
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Callable, Dict, Iterable, Mapping, Set, Tuple

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
from ...core import Evaluator, Runner, RunnerCallback, Trainer
from ...game import (
    IDecoder,
    IEncoder,
    IEncoderDecoder,
    ReconstructionNetworkGame,
    ReconstructionNetworkSubGame,
    ReconstructionNetworkSubGameResult,
)
from ...loggers import WandbLogger
from ...loss import ReinforceLoss
from ...utils import (
    DuplicateChecker,
    EarlyStopper,
    Interval,
    StepCounter,
    Timer,
    fix_seed,
    init_weights,
    random_split,
)
from .agent import Agent, MessageAuxiliary
from .loss import Loss
from .metrics import Metrics, TopographicSimilarityMetrics


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
        agents=agents.values(),
        input=train_dataloader,
        games=[game],
        loss=loss,
        optimizers=optimizers.values(),
    )

    wandb_logger = WandbLogger(project=cfg.wandb_project, name=run_name)
    duplicate_checker = DuplicateChecker()

    adj_all: Dict[str, Set[str]] = {s: {t for t in agents} for s in agents}
    adj_nil: Dict[str, Set[str]] = {s: set() for s in agents}
    game_all: ReconstructionNetworkGame[
        TensorType[..., int],
        TensorType[..., int],
        MessageAuxiliary,
        TensorType[..., float],
    ] = ReconstructionNetworkGame(agents, adj_all)
    game_nil: ReconstructionNetworkGame[
        TensorType[..., int],
        TensorType[..., int],
        MessageAuxiliary,
        TensorType[..., float],
    ] = ReconstructionNetworkGame(agents, adj_nil)

    metrics_evals = []
    topsim_evals = []
    for name, input in [
        ("train", train_dataloader),
        ("test", test_dataloader),
    ]:
        metrics = Metrics(
            name=name, loss=loss, callbacks=[wandb_logger, duplicate_checker]
        )
        metrics_evals.append(
            Evaluator(
                agents=agents.values(),
                input=input,
                games=[game_all],
                callbacks=[metrics],
            )
        )

        topsim = TopographicSimilarityMetrics(
            name=name, callbacks=[wandb_logger, duplicate_checker]
        )
        topsim_evals.append(
            Evaluator(
                agents=agents.values(),
                input=input,
                games=[game_nil],
                callbacks=[topsim],
            )
        )

    runner_callbacks = [
        trainer,
        Interval(cfg.metrics_interval, metrics_evals),
        Interval(cfg.topsim_interval, topsim_evals),
        StepCounter("step", [wandb_logger, duplicate_checker]),
        wandb_logger,
        duplicate_checker,
    ]
    if game_editor is not None:
        runner_callbacks = [game_editor] + runner_callbacks
    # runner_callbacks = [Timer(runner_callbacks)]
    runner = Runner(runner_callbacks)
    runner.run(cfg.n_epochs)
