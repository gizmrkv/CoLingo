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
from ...game import IDecoder, IEncoder, ReconstructionGame, ReconstructionGameResult
from ...loggers import WandbLogger
from ...utils import (
    DuplicateChecker,
    Interval,
    StepCounter,
    Timer,
    fix_seed,
    init_weights,
    random_split,
)


def loss(
    step: int,
    input: TensorType[..., int],
    outputs: Iterable[
        ReconstructionGameResult[
            TensorType[..., int], TensorType[..., float], None, TensorType[..., float]
        ]
    ],
) -> TensorType[1, float]:
    return torch.stack(
        [
            F.cross_entropy(
                output.decoder_aux.view(-1, output.decoder_aux.shape[-1]),
                output.input.view(-1),
            )
            for output in outputs
        ]
    ).mean()
