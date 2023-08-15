import datetime
import json
import os
import uuid
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Callable, Iterable, Mapping, Tuple

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


@dataclass
class MessageAuxiliary:
    max_len: int
    n_values: int
    message: TensorType[..., "max_len", int]
    logits: TensorType[..., "max_len", "n_values", float]
    log_prob: TensorType[..., "max_len", float]
    entropy: TensorType[..., "max_len", float]
    length: TensorType[..., int]


class Agent(
    nn.Module,
    IEncoderDecoder[
        TensorType[..., int],
        TensorType[..., int],
        MessageAuxiliary,
        TensorType[..., float],
    ],
):
    def __init__(
        self,
        object_encoder: nn.Module,
        object_decoder: nn.Module,
        message_encoder: nn.Module,
        message_decoder: nn.Module,
        eos: int = 0,
    ) -> None:
        super().__init__()
        self.object_encoder = object_encoder
        self.object_decoder = object_decoder
        self.message_encoder = message_encoder
        self.message_decoder = message_decoder
        self.eos = eos

    def encode(
        self, input: TensorType[..., int]
    ) -> Tuple[TensorType[..., int], MessageAuxiliary]:
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

    def decode(
        self, latent: TensorType[..., float]
    ) -> Tuple[TensorType[..., int], TensorType[..., float]]:
        latent = self.message_encoder(latent)
        output, aux = self.object_decoder(latent)
        return output, aux
