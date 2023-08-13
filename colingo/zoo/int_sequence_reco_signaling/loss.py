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
from ...game import IDecoder, IEncoder, ReconstructionGame, ReconstructionGameResult
from ...loggers import IntSequenceLanguageLogger, WandbLogger
from ...loss import ReinforceLoss
from .agent import MessageAuxiliary


class Loss:
    def __init__(
        self,
        object_length: int,
        object_n_values: int,
        message_length: int,
        message_n_values: int,
        entropy_weight: float = 0.0,
        length_weight: float = 0.0,
        baseline: Callable[[TensorType[..., float]], TensorType[..., float]]
        | None = None,
        length_baseline: Callable[[TensorType[..., float]], TensorType[..., float]]
        | None = None,
    ) -> None:
        super().__init__()
        self.object_length = object_length
        self.object_n_values = object_n_values
        self.message_length = message_length
        self.message_n_values = message_n_values
        self.entropy_weight = entropy_weight
        self.length_weight = length_weight
        self.baseline = baseline
        self.length_baseline = length_baseline

        self.reinforce_loss = ReinforceLoss(
            max_len=message_length,
            entropy_weight=entropy_weight,
            length_weight=length_weight,
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
                result.decoder_aux.view(-1, self.object_n_values),
                result.input.view(-1),
                reduction="none",
            )
            .view(-1, self.object_length)
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

    def total_loss(
        self,
        output: ReconstructionGameResult[
            TensorType[..., int],
            TensorType[..., int],
            TensorType[..., float],
            TensorType[..., float],
        ],
    ) -> TensorType[..., float]:
        decoder_loss = self.decoder_loss(output)
        encoder_loss = self.encoder_loss(output, decoder_loss)
        return decoder_loss + encoder_loss

    def __call__(
        self,
        step: int,
        input: TensorType[..., int],
        outputs: Iterable[
            ReconstructionGameResult[
                TensorType[..., int],
                TensorType[..., int],
                TensorType[..., float],
                TensorType[..., float],
            ]
        ],
    ) -> TensorType[1, float]:
        return torch.stack([self.total_loss(output) for output in outputs]).mean()
