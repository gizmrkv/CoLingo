import datetime
import json
import os
import uuid
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Callable, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtyping import TensorType

from ...core import Evaluator, Runner, Trainer
from ...game.reconstruction import (
    IDecoder,
    IEncoder,
    ReconstructionGame,
    ReconstructionGameResult,
)
from ...loggers import WandbLogger
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


@dataclass
class Config:
    n_epochs: int
    batch_size: int
    device: str
    seed: int
    wandb_project: str
    use_tqdm: bool

    lr: float
    length: int
    n_values: int

    metrics_interval: int


class Encoder(nn.Module, IEncoder[TensorType[..., int], TensorType[..., float], None]):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def encode(
        self, input: TensorType[..., int]
    ) -> Tuple[TensorType[..., float], None]:
        return self.model(input), None


class Decoder(
    nn.Module,
    IDecoder[TensorType[..., float], TensorType[..., int], TensorType[..., float]],
):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def decode(
        self, latent: TensorType[..., float]
    ) -> Tuple[TensorType[..., int], TensorType[..., float]]:
        return self.model(latent)  # type: ignore


def loss(
    result: ReconstructionGameResult[
        TensorType[..., int], TensorType[..., float], None, TensorType[..., float]
    ]
) -> TensorType[..., float]:
    return F.cross_entropy(
        result.decoder_aux.view(-1, result.decoder_aux.shape[-1]),
        result.input.view(-1),
    )


class Metrics:
    def __init__(
        self,
        name: str,
        length: int,
        n_values: int,
        callbacks: Iterable[Callable[[dict[str, float]], None]],
    ) -> None:
        self.name = name
        self.length = length
        self.n_values = n_values
        self.callbacks = callbacks

    def __call__(
        self,
        result: ReconstructionGameResult[
            TensorType[..., "length", int],
            TensorType[..., float],
            None,
            TensorType[..., "length", "n_values", float],
        ],
    ) -> None:
        metrics: dict[str, float] = {}

        mark = result.output == result.input
        acc_comp = mark.all(dim=-1).float().mean().item()
        acc = mark.float().mean(dim=0)
        acc_part = acc.mean().item()
        metrics |= {"acc_comp": acc_comp, "acc_part": acc_part}
        metrics |= {f"acc{i}": a.item() for i, a in enumerate(list(acc))}

        metrics["loss"] = loss(result).mean().item()

        metrics = {f"{self.name}.{k}": v for k, v in metrics.items()}
        for callback in self.callbacks:
            callback(metrics)


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
        torch.Tensor(list(product(torch.arange(cfg.n_values), repeat=cfg.length)))
        .long()
        .to(cfg.device)
    )
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True  # type: ignore
    )
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size)  # type: ignore

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
    early_stopper = EarlyStopper(
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
        game=game,
        callbacks=[train_metrics],
    )
    test_evaluator = Evaluator(
        agents=models,
        input=test_dataloader,
        game=game,
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
