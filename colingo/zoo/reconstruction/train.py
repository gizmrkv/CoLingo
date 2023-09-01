import datetime
import json
import os
import uuid
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, List, Literal, Mapping

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ...core import Evaluator, Runner, RunnerCallback, Trainer
from ...game import ReconstructionGame
from ...loggers import WandbLogger
from ...module import (
    MLPDecoder,
    MLPEncoder,
    RNNDecoder,
    RNNEncoder,
    TransformerDecoder,
    TransformerEncoder,
)
from ...utils import (
    DuplicateChecker,
    MetricsEarlyStopper,
    StepCounter,
    Stopwatch,
    Timer,
    fix_seed,
    init_weights,
    random_split,
)
from .agent import Decoder, Encoder
from .loss import loss
from .metrics import Metrics


def train_reconstruction(
    encoder: Encoder,
    decoder: Decoder,
    length: int,
    values: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    wandb_project: str,
    use_tqdm: bool,
    metrics_interval: int,
    additions: Iterable[RunnerCallback] | None = None,
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
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    game = ReconstructionGame(encoder, decoder)
    trainer = Trainer(
        agents=models,
        input=train_dataloader,
        games=[game],
        loss=loss,
        optimizers=optimizers,
    )

    wandb_logger = WandbLogger(project=wandb_project)
    duplicate_checker = DuplicateChecker()
    early_stopper = MetricsEarlyStopper(
        lambda metrics: metrics["test.acc_comp"] > 0.99
        if "test.acc_comp" in metrics
        else False
    )
    evaluators = []
    for name, input in [
        ("train", train_dataloader),
        ("test", test_dataloader),
    ]:
        evaluators.append(
            Evaluator(
                agents=models,
                input=input,
                games=[game],
                callbacks=[
                    Metrics(name, [wandb_logger, early_stopper, duplicate_checker])
                ],
                intervals=[metrics_interval],
            )
        )

    runner_callbacks = [
        *(additions or []),
        trainer,
        *evaluators,
        StepCounter("step", [wandb_logger, duplicate_checker]),
        Stopwatch("time", [wandb_logger, duplicate_checker]),
        wandb_logger,
        early_stopper,
        duplicate_checker,
    ]
    # runner_callbacks = [Timer(runner_callbacks)]
    runner = Runner(runner_callbacks, stopper=early_stopper, use_tqdm=use_tqdm)
    runner.run(n_epochs)


@dataclass
class ReconstructionConfig:
    length: int
    values: int
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
    config: Mapping[str, Any], additions: Iterable[RunnerCallback] | None = None
) -> None:
    fields = ReconstructionConfig.__dataclass_fields__
    config = {k: v for k, v in config.items() if k in fields}
    cfg = ReconstructionConfig(**config)

    if cfg.encoder_type == "mlp":
        encoder = Encoder(
            MLPEncoder(
                max_len=cfg.length,
                vocab_size=cfg.values,
                output_dim=cfg.latent_dim,
                **cfg.encoder_params,
            )
        )
    elif cfg.encoder_type == "rnn":
        encoder = Encoder(
            RNNEncoder(
                vocab_size=cfg.values,
                output_dim=cfg.latent_dim,
                **cfg.encoder_params,
            )
        )
    elif cfg.encoder_type == "transformer":
        encoder = Encoder(
            TransformerEncoder(
                vocab_size=cfg.values,
                output_dim=cfg.latent_dim,
                **cfg.encoder_params,
            )
        )

    if cfg.decoder_type == "mlp":
        decoder = Decoder(
            MLPDecoder(
                max_len=cfg.length,
                vocab_size=cfg.values,
                input_dim=cfg.latent_dim,
                **cfg.decoder_params,
            )
        )
    elif cfg.decoder_type == "rnn":
        decoder = Decoder(
            RNNDecoder(
                input_dim=cfg.latent_dim,
                max_len=cfg.length,
                vocab_size=cfg.values,
                **cfg.decoder_params,
            )
        )
    elif cfg.decoder_type == "transformer":
        decoder = Decoder(
            TransformerDecoder(
                input_dim=cfg.latent_dim,
                max_len=cfg.length,
                vocab_size=cfg.values,
                **cfg.decoder_params,
            )
        )

    train_reconstruction(
        encoder=encoder,
        decoder=decoder,
        length=cfg.length,
        values=cfg.values,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=cfg.device,
        wandb_project=cfg.wandb_project,
        use_tqdm=cfg.use_tqdm,
        metrics_interval=cfg.metrics_interval,
        additions=additions,
    )
