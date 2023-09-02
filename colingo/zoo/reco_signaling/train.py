import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtyping import TensorType

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
from .game import RecoSignalingGame
from .loss import Loss
from .metrics import LanguageLoggerWrapper, Metrics, TopographicSimilarityMetrics


def train_reco_signaling(
    concept_encoder: nn.Module,
    message_decoder: nn.Module,
    message_encoder: nn.Module,
    concept_decoder: nn.Module,
    concept_length: int,
    concept_values: int,
    train_proportion: float,
    valid_proportion: float,
    message_max_len: int,
    message_vocab_size: int,
    entropy_weight: float,
    length_weight: float,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    wandb_project: str,
    use_tqdm: bool,
    metrics_interval: int,
    topsim_interval: int,
    lang_log_interval: int,
    log_dir: Path,
    additions: Iterable[Task] | None = None,
) -> None:
    models: List[nn.Module] = [
        concept_encoder,
        message_decoder,
        message_encoder,
        concept_decoder,
    ]
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in models]

    for model in models:
        model.to(device)
        model.apply(init_weights)

    dataset = (
        torch.Tensor(list(product(torch.arange(concept_values), repeat=concept_length)))
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

    game = RecoSignalingGame(
        concept_encoder, message_decoder, message_encoder, concept_decoder
    )

    def baseline(x: TensorType[..., float]) -> TensorType[..., float]:
        return x.detach().mean(dim=0)

    loss = Loss(
        concept_length=concept_length,
        concept_values=concept_values,
        message_max_len=message_max_len,
        message_vocab_size=message_vocab_size,
        entropy_weight=entropy_weight,
        length_weight=length_weight,
        baseline=baseline,
        length_baseline=baseline,
    )

    trainer = Trainer(
        agents=models,
        input=train_dataloader,
        game=game,
        loss=loss,
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
        metrics = Metrics(loss=loss, loggers=[Namer(name, loggers)])

        topsim = TopographicSimilarityMetrics(loggers=[Namer(name, loggers)])
        evaluators.append(
            Evaluator(
                agents=models,
                input=input,
                game=game,
                metrics=[metrics, topsim],
                intervals=[metrics_interval, topsim_interval],
            )
        )

    language_logger = LanguageLoggerWrapper(log_dir.joinpath("lang"))
    evaluators.append(
        Evaluator(
            agents=models,
            input=DataLoader(dataset, batch_size=len(dataset)),  # type: ignore
            game=game,
            metrics=[language_logger],
            intervals=[lang_log_interval],
        )
    )

    runner_callbacks = [
        *(additions or []),
        trainer,
        *evaluators,
        StepCounter(loggers),
        Stopwatch(loggers),
        wandb_logger,
        early_stopper,
        key_checker,
    ]
    # runner_callbacks = [Timer(runner_callbacks)]
    runner = TaskRunner(runner_callbacks, stopper=early_stopper, use_tqdm=use_tqdm)
    runner.run(n_epochs)


@dataclass
class RecoSignalingConfig:
    concept_length: int
    concept_values: int
    train_proportion: float
    valid_proportion: float
    message_max_len: int
    message_vocab_size: int
    entropy_weight: float
    length_weight: float
    n_epochs: int
    batch_size: int
    lr: float
    device: str
    wandb_project: str
    use_tqdm: bool
    metrics_interval: int
    topsim_interval: int
    language_log_interval: int

    sender_latent_dim: int
    sender_encoder_type: Literal["mlp", "rnn", "transformer"]
    sender_encoder_params: Mapping[str, Any]
    sender_decoder_type: Literal["mlp", "rnn", "transformer"]
    sender_decoder_params: Mapping[str, Any]

    receiver_latent_dim: int
    receiver_encoder_type: Literal["mlp", "rnn", "transformer"]
    receiver_encoder_params: Mapping[str, Any]
    receiver_decoder_type: Literal["mlp", "rnn", "transformer"]
    receiver_decoder_params: Mapping[str, Any]


def train_reco_signaling_from_config(
    config: Mapping[str, Any],
    log_dir: Path,
    additions: Iterable[Task] | None = None,
) -> None:
    fields = RecoSignalingConfig.__dataclass_fields__
    config = {k: v for k, v in config.items() if k in fields}
    cfg = RecoSignalingConfig(**config)

    if cfg.sender_encoder_type == "mlp":
        sender_encoder: nn.Module = MLPEncoder(
            max_len=cfg.concept_length,
            vocab_size=cfg.concept_values,
            output_dim=cfg.sender_latent_dim,
            **cfg.sender_encoder_params,
        )
    elif cfg.sender_encoder_type == "rnn":
        sender_encoder = RNNEncoder(
            vocab_size=cfg.concept_values,
            output_dim=cfg.sender_latent_dim,
            **cfg.sender_encoder_params,
        )
    elif cfg.sender_encoder_type == "transformer":
        sender_encoder = TransformerEncoder(
            vocab_size=cfg.concept_values,
            output_dim=cfg.sender_latent_dim,
            **cfg.sender_encoder_params,
        )

    if cfg.sender_decoder_type == "mlp":
        sender_decoder: nn.Module = MLPDecoder(
            input_dim=cfg.sender_latent_dim,
            max_len=cfg.concept_length,
            vocab_size=cfg.concept_values,
            **cfg.sender_decoder_params,
        )
    elif cfg.sender_decoder_type == "rnn":
        sender_decoder = RNNDecoder(
            input_dim=cfg.sender_latent_dim,
            max_len=cfg.message_max_len,
            vocab_size=cfg.message_vocab_size,
            **cfg.sender_decoder_params,
        )
    elif cfg.sender_decoder_type == "transformer":
        sender_decoder = TransformerDecoder(
            input_dim=cfg.sender_latent_dim,
            max_len=cfg.message_max_len,
            vocab_size=cfg.message_vocab_size,
            **cfg.sender_decoder_params,
        )

    if cfg.receiver_encoder_type == "mlp":
        receiver_encoder: nn.Module = MLPEncoder(
            max_len=cfg.message_max_len,
            vocab_size=cfg.message_vocab_size,
            output_dim=cfg.receiver_latent_dim,
            **cfg.receiver_encoder_params,
        )
    elif cfg.receiver_encoder_type == "rnn":
        receiver_encoder = RNNEncoder(
            vocab_size=cfg.message_vocab_size,
            output_dim=cfg.receiver_latent_dim,
            **cfg.receiver_encoder_params,
        )
    elif cfg.receiver_encoder_type == "transformer":
        receiver_encoder = TransformerEncoder(
            vocab_size=cfg.message_vocab_size,
            output_dim=cfg.receiver_latent_dim,
            **cfg.receiver_encoder_params,
        )

    if cfg.receiver_decoder_type == "mlp":
        receiver_decoder: nn.Module = MLPDecoder(
            input_dim=cfg.receiver_latent_dim,
            max_len=cfg.concept_length,
            vocab_size=cfg.concept_values,
            **cfg.receiver_decoder_params,
        )
    elif cfg.receiver_decoder_type == "rnn":
        receiver_decoder = RNNDecoder(
            input_dim=cfg.receiver_latent_dim,
            max_len=cfg.concept_length,
            vocab_size=cfg.concept_values,
            **cfg.receiver_decoder_params,
        )
    elif cfg.receiver_decoder_type == "transformer":
        receiver_decoder = TransformerDecoder(
            input_dim=cfg.receiver_latent_dim,
            max_len=cfg.concept_length,
            vocab_size=cfg.concept_values,
            **cfg.receiver_decoder_params,
        )

    train_reco_signaling(
        concept_encoder=sender_encoder,
        message_decoder=sender_decoder,
        message_encoder=receiver_encoder,
        concept_decoder=receiver_decoder,
        concept_length=cfg.concept_length,
        concept_values=cfg.concept_values,
        train_proportion=cfg.train_proportion,
        valid_proportion=cfg.valid_proportion,
        message_max_len=cfg.message_max_len,
        message_vocab_size=cfg.message_vocab_size,
        entropy_weight=cfg.entropy_weight,
        length_weight=cfg.length_weight,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=cfg.device,
        wandb_project=cfg.wandb_project,
        use_tqdm=cfg.use_tqdm,
        metrics_interval=cfg.metrics_interval,
        topsim_interval=cfg.topsim_interval,
        lang_log_interval=cfg.language_log_interval,
        log_dir=log_dir,
        additions=additions,
    )
