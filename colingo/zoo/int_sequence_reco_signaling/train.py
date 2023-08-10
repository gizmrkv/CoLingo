import datetime
import json
import os
import uuid
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Callable, Iterable, Tuple

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
from ...core import Evaluator, Runner, Trainer
from ...game.reconstruction import (
    IDecoder,
    IEncoder,
    ReconstructionGame,
    ReconstructionGameResult,
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


@dataclass
class Config:
    n_epochs: int
    batch_size: int
    device: str
    seed: int
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


@dataclass
class MessageAuxiliary:
    max_len: int
    n_values: int
    message: TensorType[..., "max_len", int]
    logits: TensorType[..., "max_len", "n_values", float]
    log_prob: TensorType[..., "max_len", float]
    entropy: TensorType[..., "max_len", float]
    length: TensorType[..., int]


class Encoder(
    nn.Module,
    IEncoder[TensorType[..., int], TensorType[..., int], MessageAuxiliary],
):
    def __init__(
        self, object_encoder: nn.Module, message_decoder: nn.Module, eos: int = 0
    ):
        super().__init__()
        self.object_encoder = object_encoder
        self.message_decoder = message_decoder
        self.eos = eos

    def encode(
        self, input: TensorType[..., int]
    ) -> Tuple[TensorType[..., int], TensorType[..., float]]:
        latent = self.object_encoder(input)
        message, logits = self.message_decoder(latent)

        distr = Categorical(logits=logits)
        log_prob = distr.log_prob(message)
        entropy = distr.entropy()

        mask = message == self.eos
        indices = torch.argmax(mask.int(), dim=1)
        no_mask = ~mask.any(dim=1)
        indices[no_mask] = message.shape[1]
        mask = torch.arange(message.shape[1]).expand(message.shape).to(message.device)
        mask = (mask <= indices.unsqueeze(-1)).long()

        length = mask.sum(dim=-1)
        message = message * mask
        log_prob = log_prob * mask
        entropy = entropy * mask

        return message, MessageAuxiliary(
            max_len=message.shape[1],
            n_values=logits.shape[-1],
            message=message,
            logits=logits,
            log_prob=log_prob,
            entropy=entropy,
            length=length,
        )


class Decoder(
    nn.Module,
    IDecoder[TensorType[..., int], TensorType[..., int], TensorType[..., float]],
):
    def __init__(self, message_encoder: nn.Module, object_decoder: nn.Module):
        super().__init__()
        self.message_encoder = message_encoder
        self.object_decoder = object_decoder

    def decode(
        self, latent: TensorType[..., float]
    ) -> Tuple[TensorType[..., int], TensorType[..., float]]:
        latent = self.message_encoder(latent)
        output, aux = self.object_decoder(latent)
        return output, aux


class Loss:
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        def baseline(x: TensorType[..., float]) -> TensorType[..., float]:
            return x.detach().mean(dim=0)

        self.reinforce_loss = ReinforceLoss(
            max_len=cfg.message_length,
            entropy_weight=cfg.entropy_weight,
            length_weight=cfg.length_weight,
            baseline=baseline,
            length_baseline=baseline,
        )

    def decoder_loss(
        self,
        result: ReconstructionGameResult[
            TensorType[..., int],
            TensorType[..., int],
            TensorType[..., float],
            TensorType[..., float],
        ],
    ) -> TensorType[..., float]:
        return (
            F.cross_entropy(
                result.decoder_aux.view(-1, self.cfg.object_n_values),
                result.input.view(-1),
                reduction="none",
            )
            .view(-1, self.cfg.object_length)
            .mean(dim=-1)
        )

    def encoder_loss(
        self,
        result: ReconstructionGameResult[
            TensorType[..., int],
            TensorType[..., int],
            MessageAuxiliary,
            TensorType[..., float],
        ],
        decoder_loss: TensorType[..., float],
    ) -> TensorType[..., float]:
        return self.reinforce_loss(
            reward=-decoder_loss.detach(),
            log_prob=result.encoder_aux.log_prob,
            entropy=result.encoder_aux.entropy,
            length=result.encoder_aux.length,
        )

    def __call__(
        self,
        result: ReconstructionGameResult[
            TensorType[..., int],
            TensorType[..., int],
            TensorType[..., float],
            TensorType[..., float],
        ],
    ) -> TensorType[..., float]:
        decoder_loss = self.decoder_loss(result)
        encoder_loss = self.encoder_loss(result, decoder_loss)
        return decoder_loss + encoder_loss


class Metrics:
    def __init__(
        self,
        name: str,
        object_length: int,
        object_n_values: int,
        message_length: int,
        message_n_values: int,
        loss: Loss,
        callbacks: Iterable[Callable[[dict[str, float]], None]],
    ) -> None:
        self.name = name
        self.object_length = object_length
        self.object_n_values = object_n_values
        self.message_length = message_length
        self.message_n_values = message_n_values
        self.loss = loss
        self.callbacks = callbacks

    def __call__(
        self,
        result: ReconstructionGameResult[
            TensorType[..., "object_length", int],
            TensorType[..., "message_length", int],
            MessageAuxiliary,
            TensorType[..., "object_length", "object_n_values", float],
        ],
    ) -> None:
        metrics: dict[str, float] = {}

        mark = result.output == result.input
        acc_comp = mark.all(dim=-1).float().mean().item()
        acc = mark.float().mean(dim=0)
        acc_part = acc.mean().item()
        metrics |= {"acc_comp": acc_comp, "acc_part": acc_part}
        metrics |= {f"acc{i}": a.item() for i, a in enumerate(list(acc))}

        metrics |= {
            "entropy": result.encoder_aux.entropy.mean().item(),
            "length": result.encoder_aux.length.float().mean().item(),
            "unique": result.encoder_aux.message.unique(dim=0).shape[0]
            / result.encoder_aux.message.shape[0],
        }

        decoder_loss = self.loss.decoder_loss(result)
        encoder_loss = self.loss.encoder_loss(result, decoder_loss)
        total_loss = decoder_loss + encoder_loss
        metrics |= {
            "decoder_loss": decoder_loss.mean().item(),
            "encoder_loss": encoder_loss.mean().item(),
            "total_loss": total_loss.mean().item(),
        }

        metrics = {f"{self.name}.{k}": v for k, v in metrics.items()}
        for callback in self.callbacks:
            callback(metrics)


def drop_padding(x: NDArray[np.int32]) -> NDArray[np.int32]:
    i = np.argwhere(x == 0)
    return x if len(i) == 0 else x[: i[0, 0]]


class TopographicSimilarityMetrics:
    def __init__(
        self, name: str, callbacks: Iterable[Callable[[dict[str, float]], None]]
    ) -> None:
        self.name = name
        self.callbacks = callbacks

    def __call__(
        self,
        result: ReconstructionGameResult[
            TensorType[..., int],
            TensorType[..., int],
            MessageAuxiliary,
            TensorType[..., float],
        ],
    ) -> None:
        metrics = {
            f"{self.name}.topsim": topographic_similarity(
                result.input.cpu().numpy(),
                result.latent.cpu().numpy(),
                y_processor=drop_padding,  # type: ignore
            )
        }

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

    game = ReconstructionGame(encoder, decoder)
    loss = Loss(cfg)
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
        name="train",
        object_length=cfg.object_length,
        object_n_values=cfg.object_n_values,
        message_length=cfg.message_length,
        message_n_values=cfg.message_n_values,
        loss=loss,
        callbacks=[wandb_logger, duplicate_checker],
    )
    test_metrics = Metrics(
        name="test",
        object_length=cfg.object_length,
        object_n_values=cfg.object_n_values,
        message_length=cfg.message_length,
        message_n_values=cfg.message_n_values,
        loss=loss,
        callbacks=[wandb_logger, early_stopper, duplicate_checker],
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

    train_topsim = TopographicSimilarityMetrics(
        "train", [wandb_logger, duplicate_checker]
    )
    test_topsim = TopographicSimilarityMetrics(
        "test", [wandb_logger, duplicate_checker]
    )
    train_topsim_evaluator = Evaluator(
        agents=models,
        input=train_dataloader,
        game=game,
        callbacks=[train_topsim],
    )
    test_topsim_evaluator = Evaluator(
        agents=models,
        input=test_dataloader,
        game=game,
        callbacks=[test_topsim],
    )

    runner_callbacks = [
        trainer,
        Interval(cfg.metrics_interval, [train_evaluator, test_evaluator]),
        Interval(cfg.topsim_interval, [train_topsim_evaluator, test_topsim_evaluator]),
        StepCounter("step", [wandb_logger, duplicate_checker]),
        wandb_logger,
        early_stopper,
        duplicate_checker,
    ]
    # runner_callbacks = [Timer(runner_callbacks)]
    runner = Runner(runner_callbacks, stopper=early_stopper)
    runner.run(cfg.n_epochs)
