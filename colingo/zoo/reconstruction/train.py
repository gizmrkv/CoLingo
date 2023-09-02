import math
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, List, Literal, Mapping

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ...core import Evaluator, Loggable, Task, TaskRunner, Trainer
from ...loggers import Namer, WandbLogger
from ...module import (
    MLPDecoder,
    MLPEncoder,
    RNNDecoder,
    RNNEncoder,
    TransformerDecoder,
    TransformerEncoder,
)
from ...tasks import DictStopper, KeyChecker, StepCounter, Stopwatch, TimeDebugger
from ...utils import init_weights, random_split
from .game import ReconstructionGame
from .loss import Loss
from .metrics import Metrics


def train_reconstruction(
    encoder: nn.Module,
    decoder: nn.Module,
    length: int,
    values: int,
    train_proportion: float,
    valid_proportion: float,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    wandb_project: str,
    use_tqdm: bool,
    metrics_interval: int,
    additional_tasks: Iterable[Task] | None = None,
) -> None:
    models: List[nn.Module] = [encoder, decoder]
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in models]

    for model in [encoder, decoder]:
        model.to(device)
        model.apply(init_weights)

    dataset = (
        torch.Tensor(list(product(torch.arange(values), repeat=length)))
        .long()
        .to(device)
    )
    train_dataset, valid_dataset = random_split(
        dataset, [train_proportion, valid_proportion]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=len(valid_dataset), shuffle=False
    )

    game = ReconstructionGame(encoder, decoder)
    trainer = Trainer(
        agents=models,
        input=train_dataloader,
        game=game,
        loss=Loss(),
        optimizers=optimizers,
    )

    wandb_logger = WandbLogger(project=wandb_project)
    key_checker = KeyChecker()
    early_stopper = DictStopper(
        lambda metrics: math.isclose(metrics["valid.acc_part"], 1.0)
        if "valid.acc_part" in metrics
        else False
    )
    loggers: List[Loggable[Mapping[str, Any]]] = [
        wandb_logger,
        key_checker,
        early_stopper,
    ]
    evaluators = []
    for name, input in [
        ("train", train_dataloader),
        ("valid", valid_dataloader),
    ]:
        evaluators.append(
            Evaluator(
                agents=models,
                input=input,
                game=game,
                metrics=[Metrics(Loss(), [Namer(name, loggers)])],
                intervals=[metrics_interval],
            )
        )

    runner_callbacks = [
        *(additional_tasks or []),
        trainer,
        *evaluators,
        StepCounter(loggers),
        Stopwatch(loggers),
        wandb_logger,
        key_checker,
        early_stopper,
    ]
    # runner_callbacks = [TimeDebugger(runner_callbacks)]
    runner = TaskRunner(runner_callbacks, stopper=early_stopper, use_tqdm=use_tqdm)
    runner.run(n_epochs)


@dataclass
class ReconstructionConfig:
    length: int
    values: int
    train_proportion: float
    valid_proportion: float
    n_epochs: int
    batch_size: int
    lr: float
    device: str
    wandb_project: str
    use_tqdm: bool
    metrics_interval: int

    latent_dim: int
    encoder_type: Literal["mlp", "rnn", "transformer"]
    encoder_params: Mapping[str, Any]
    decoder_type: Literal["mlp", "rnn", "transformer"]
    decoder_params: Mapping[str, Any]


def train_reconstruction_from_config(
    config: Mapping[str, Any], additional_tasks: Iterable[Task] | None = None
) -> None:
    fields = ReconstructionConfig.__dataclass_fields__
    config = {k: v for k, v in config.items() if k in fields}
    cfg = ReconstructionConfig(**config)

    if cfg.encoder_type == "mlp":
        encoder: nn.Module = MLPEncoder(
            max_len=cfg.length,
            vocab_size=cfg.values,
            output_dim=cfg.latent_dim,
            **cfg.encoder_params,
        )
    elif cfg.encoder_type == "rnn":
        encoder = RNNEncoder(
            vocab_size=cfg.values,
            output_dim=cfg.latent_dim,
            **cfg.encoder_params,
        )
    elif cfg.encoder_type == "transformer":
        encoder = TransformerEncoder(
            vocab_size=cfg.values,
            output_dim=cfg.latent_dim,
            **cfg.encoder_params,
        )

    if cfg.decoder_type == "mlp":
        decoder: nn.Module = MLPDecoder(
            max_len=cfg.length,
            vocab_size=cfg.values,
            input_dim=cfg.latent_dim,
            **cfg.decoder_params,
        )
    elif cfg.decoder_type == "rnn":
        decoder = RNNDecoder(
            input_dim=cfg.latent_dim,
            max_len=cfg.length,
            vocab_size=cfg.values,
            **cfg.decoder_params,
        )
    elif cfg.decoder_type == "transformer":
        decoder = TransformerDecoder(
            input_dim=cfg.latent_dim,
            max_len=cfg.length,
            vocab_size=cfg.values,
            **cfg.decoder_params,
        )

    train_reconstruction(
        encoder=encoder,
        decoder=decoder,
        length=cfg.length,
        values=cfg.values,
        train_proportion=cfg.train_proportion,
        valid_proportion=cfg.valid_proportion,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=cfg.device,
        wandb_project=cfg.wandb_project,
        use_tqdm=cfg.use_tqdm,
        metrics_interval=cfg.metrics_interval,
        additional_tasks=additional_tasks,
    )
