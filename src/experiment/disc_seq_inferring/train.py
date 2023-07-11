import datetime
import json
import os
import uuid
from dataclasses import dataclass
from itertools import product

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from ...baseline import BatchMeanBaseline
from ...core import Evaluator, Runner, Trainer, fix_seed, init_weights
from ...dataset import random_split
from ...game import InferringGame, InferringGameResult
from ...logger import EarlyStopper, Group, Logger, StepCounter, WandBLogger
from ...loss import DiscSeqReinforceLoss


@dataclass
class Config:
    # exp config
    n_epochs: int
    batch_size: int
    seed: int
    device: str
    exp_name: str
    wandb_project: str
    use_tqdm: bool

    # common config
    lr: float
    length: int
    n_values: int

    # optional config
    reinforce: bool = False
    baseline: str = "batch_mean"
    entropy_weight: float = 0.0


def main(agent: nn.Module, config: dict):
    # pre process
    cfg: Config = Config(**config)

    assert cfg.device in ["cpu", "cuda"], "Invalid device"
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Use CPU instead.")
        cfg.device = "cpu"

    now = datetime.datetime.now()
    log_id = str(uuid.uuid4())[-4:]
    log_dir = f"log/{cfg.exp_name}_{now.date()}_{now.strftime('%H%M%S')}_{log_id}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    fix_seed(cfg.seed)

    agent.apply(init_weights)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr)

    # data
    dataset = (
        torch.Tensor(list(product(torch.arange(cfg.n_values), repeat=cfg.length)))
        .long()
        .to(cfg.device)
    )
    train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    # game
    if cfg.reinforce:
        baselines = {"batch_mean": BatchMeanBaseline()}
        reinforce_loss = DiscSeqReinforceLoss(
            entropy_weight=cfg.entropy_weight, baseline=baselines[cfg.baseline]
        )

    def loss(result: InferringGameResult):
        if cfg.reinforce:
            acc = (result.output == result.input).float().mean(dim=-1)
            distr = Categorical(logits=result.info)
            log_prob = distr.log_prob(result.output)
            entropy = distr.entropy()
            return reinforce_loss(acc, log_prob, entropy)
        else:
            logits = result.info.view(-1, cfg.n_values)
            return F.cross_entropy(logits, result.input.view(-1))

    game = InferringGame(agent)
    trainer = Trainer(game, [optimizer], train_dataloader, loss)

    # eval
    class Metrics:
        def __init__(self, name: str):
            self._name = name

        def __call__(self, results: list[InferringGameResult]):
            result = results[0]
            mark = result.output == result.input
            acc_comp = mark.all(dim=-1).float().mean().item()
            acc = mark.float().mean(dim=0)
            acc_part = acc.mean().item()
            metrics = {"acc_comp": acc_comp, "acc_part": acc_part}
            metrics |= {f"acc{i}": a for i, a in enumerate(list(acc))}

            metrics = {f"{self._name}.{k}": v for k, v in metrics.items()}

            return metrics

    wandb_logger = WandBLogger(project=cfg.wandb_project)
    early_stopper = EarlyStopper(metric="valid.acc_comp", threshold=1 - 1e-6)

    train_eval = Evaluator(game, train_dataloader, Metrics("train"), wandb_logger)
    valid_eval = Evaluator(
        game, valid_dataloader, Metrics("valid"), [wandb_logger, early_stopper]
    )

    runner = Runner(
        trainer,
        train_eval,
        valid_eval,
        StepCounter(wandb_logger),
        wandb_logger,
        early_stop=early_stopper,
        use_tqdm=cfg.use_tqdm,
    )
    runner.run(cfg.n_epochs)
