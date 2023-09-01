from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtyping import TensorType

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
from .agent import Receiver, Sender
from .loss import Loss
from .metrics import LangLogger, Metrics, TopographicSimilarityMetrics


def train_reco_signaling(
    sender: Sender,
    receiver: Receiver,
    object_length: int,
    object_values: int,
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
    language_log_interval: int,
    log_dir: Path,
    additions: Iterable[RunnerCallback] | None = None,
) -> None:
    models: List[nn.Module] = [sender, receiver]
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in models]

    for model in [sender, receiver]:
        model.to(device)
        model.apply(init_weights)

    dataset = (
        torch.Tensor(list(product(torch.arange(object_values), repeat=object_length)))
        .long()
        .to(device)
    )
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

    game = ReconstructionGame(sender, receiver)

    def baseline(x: TensorType[..., float]) -> TensorType[..., float]:
        return x.detach().mean(dim=0)

    loss = Loss(
        object_length=object_length,
        object_values=object_values,
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
        metrics = Metrics(
            name=name,
            loss=loss,
            callbacks=[wandb_logger, early_stopper, duplicate_checker],
        )

        topsim = TopographicSimilarityMetrics(
            name=name, callbacks=[wandb_logger, duplicate_checker]
        )
        evaluators.append(
            Evaluator(
                agents=models,
                input=input,
                games=[game],
                callbacks=[metrics, topsim],
                intervals=[metrics_interval, topsim_interval],
            )
        )

    language_logger = LangLogger(log_dir.joinpath("lang"))
    evaluators.append(
        Evaluator(
            agents=models,
            input=DataLoader(dataset, batch_size=len(dataset)),  # type: ignore
            games=[game],
            callbacks=[language_logger],
            intervals=[language_log_interval],
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
class RecoSignalingConfig:
    object_length: int
    object_values: int
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
    additions: Iterable[RunnerCallback] | None = None,
) -> None:
    fields = RecoSignalingConfig.__dataclass_fields__
    config = {k: v for k, v in config.items() if k in fields}
    cfg = RecoSignalingConfig(**config)

    if cfg.sender_encoder_type == "mlp":
        sender_encoder: nn.Module = MLPEncoder(
            max_len=cfg.object_length,
            vocab_size=cfg.object_values,
            output_dim=cfg.sender_latent_dim,
            **cfg.sender_encoder_params,
        )
    elif cfg.sender_encoder_type == "rnn":
        sender_encoder = RNNEncoder(
            vocab_size=cfg.object_values,
            output_dim=cfg.sender_latent_dim,
            **cfg.sender_encoder_params,
        )
    elif cfg.sender_encoder_type == "transformer":
        sender_encoder = TransformerEncoder(
            vocab_size=cfg.object_values,
            output_dim=cfg.sender_latent_dim,
            **cfg.sender_encoder_params,
        )

    if cfg.sender_decoder_type == "mlp":
        sender_decoder: nn.Module = MLPDecoder(
            input_dim=cfg.sender_latent_dim,
            max_len=cfg.object_length,
            vocab_size=cfg.object_values,
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
            max_len=cfg.object_length,
            vocab_size=cfg.object_values,
            **cfg.receiver_decoder_params,
        )
    elif cfg.receiver_decoder_type == "rnn":
        receiver_decoder = RNNDecoder(
            input_dim=cfg.receiver_latent_dim,
            max_len=cfg.object_length,
            vocab_size=cfg.object_values,
            **cfg.receiver_decoder_params,
        )
    elif cfg.receiver_decoder_type == "transformer":
        receiver_decoder = TransformerDecoder(
            input_dim=cfg.receiver_latent_dim,
            max_len=cfg.object_length,
            vocab_size=cfg.object_values,
            **cfg.receiver_decoder_params,
        )

    sender = Sender(sender_encoder, sender_decoder)
    receiver = Receiver(receiver_encoder, receiver_decoder)

    train_reco_signaling(
        sender=sender,
        receiver=receiver,
        object_length=cfg.object_length,
        object_values=cfg.object_values,
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
        language_log_interval=cfg.language_log_interval,
        log_dir=log_dir,
        additions=additions,
    )
