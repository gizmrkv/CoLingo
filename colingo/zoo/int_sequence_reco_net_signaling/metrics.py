import datetime
import json
import os
import uuid
from dataclasses import asdict, dataclass
from itertools import product
from statistics import fmean
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple

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


class Metrics:
    def __init__(
        self,
        name: str,
        object_length: int,
        object_n_values: int,
        message_length: int,
        message_n_values: int,
        loss: Loss,
        callbacks: Iterable[Callable[[Dict[str, float]], None]],
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
        step: int,
        input: TensorType[..., "object_length", int],
        outputs: Iterable[
            Dict[
                str,
                ReconstructionNetworkSubGameResult[
                    TensorType[..., "object_length", int],
                    TensorType[..., "message_length", int],
                    MessageAuxiliary,
                    TensorType[..., "object_length", "object_n_values", float],
                ],
            ]
        ],
    ) -> None:
        metrics: Dict[str, float] = {}

        result = next(iter(outputs))

        for name_e, result_e in result.items():
            acc_comps, acc_parts = zip(
                *[
                    self.acc(output_d, result_e.input)
                    for output_d in result_e.outputs.values()
                ]
            )
            acc_comp_mean = fmean(acc_comps)
            acc_part_mean = fmean(acc_parts)
            acc_comp_max = max(acc_comps)
            acc_part_max = max(acc_parts)
            metrics |= {
                f"{name_e}.acc_comp.mean": acc_comp_mean,
                f"{name_e}.acc_part.mean": acc_part_mean,
                f"{name_e}.acc_comp.max": acc_comp_max,
                f"{name_e}.acc_part.max": acc_part_max,
            }

        metrics = {f"{self.name}.{k}": v for k, v in metrics.items()}
        for callback in self.callbacks:
            callback(metrics)

    def acc(
        self,
        input: TensorType[..., "object_length", int],
        target: TensorType[..., "object_length", int],
    ) -> Tuple[float, float]:
        mark = target == input
        acc_comp = mark.all(dim=-1).float().mean().item()
        acc_part = mark.float().mean(dim=0).mean().item()
        return acc_comp, acc_part


class TopographicSimilarityMetrics:
    def __init__(
        self, name: str, callbacks: Iterable[Callable[[Dict[str, float]], None]]
    ) -> None:
        self.name = name
        self.callbacks = callbacks

    def __call__(
        self,
        step: int,
        input: TensorType[..., int],
        outputs: Iterable[
            Dict[
                str,
                ReconstructionNetworkSubGameResult[
                    TensorType[..., int],
                    TensorType[..., int],
                    MessageAuxiliary,
                    TensorType[..., float],
                ],
            ]
        ],
    ) -> None:
        output = next(iter(outputs))
        metrics: Dict[str, float] = {}
        for name_e, result_e in output.items():
            metrics |= {
                f"{self.name}.{name_e}.topsim": topographic_similarity(
                    result_e.input.cpu().numpy(),
                    result_e.latent.cpu().numpy(),
                    y_processor=drop_padding,  # type: ignore
                )
            }

        for callback in self.callbacks:
            callback(metrics)


def drop_padding(x: NDArray[np.int32]) -> NDArray[np.int32]:
    i = np.argwhere(x == 0)
    return x if len(i) == 0 else x[: i[0, 0]]
