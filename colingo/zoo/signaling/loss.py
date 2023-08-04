import math
from typing import Iterable, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torchtyping import TensorType

from ...baseline import BatchMeanBaseline
from ...loss import ReinforceLoss
from .game import GameResult

BATCH = "batch"
LENGTH = "length"
N_VALUES = "n_values"


def sequence_cross_entropy_loss(
    input: TensorType[BATCH, LENGTH, N_VALUES, float],
    target: TensorType[BATCH, LENGTH, int],
    length: int,
    n_values: int,
) -> TensorType[BATCH, float]:
    input = input.view(-1, n_values)
    loss = F.cross_entropy(input, target.view(-1), reduction="none")
    return loss.view(-1, length).sum(dim=-1)


def object_reinforce_loss(
    reward: TensorType[BATCH, float],
    sequence: TensorType[BATCH, LENGTH, int],
    logits: TensorType[BATCH, LENGTH, N_VALUES, float],
    reinforce_loss: ReinforceLoss,
) -> TensorType[BATCH, float]:
    distr = Categorical(logits=logits)
    log_prob = distr.log_prob(sequence)
    entropy = distr.entropy()
    return reinforce_loss(reward, log_prob, entropy)


class ReceiverObjectCrossEntropyLoss(nn.Module):
    def __init__(self, length: int, n_values: int, weight: float = 1.0):
        super().__init__()
        self._length = length
        self._n_values = n_values
        self._weight = weight

    def forward(self, result: GameResult) -> TensorType[float]:
        loss = [
            sequence_cross_entropy_loss(
                logits, result.input, self._length, self._n_values
            )
            for logits in result.output_logits_r
        ]
        loss_sum = torch.stack(loss, dim=-1)
        loss_sum = loss_sum.sum(dim=-1)
        return self._weight * loss_sum


class SenderMessageReinforceLoss(nn.Module):
    def __init__(self, reinforce_loss: ReinforceLoss, weight: float = 1.0):
        super().__init__()
        self._reinforce_loss = reinforce_loss
        self._weight = weight

    def forward(
        self, result: GameResult, reward: TensorType[BATCH, float]
    ) -> TensorType[float]:
        loss = self._reinforce_loss(
            reward.detach(),
            result.message_log_prob_s,
            result.message_entropy_s,
            result.message_length_s,
        )
        return self._weight * loss


class SenderObjectCrossEntropyLoss(nn.Module):
    def __init__(self, length: int, n_values: int, weight: float = 1.0):
        super().__init__()
        self._length = length
        self._n_values = n_values
        self._weight = weight

    def forward(self, result: GameResult) -> TensorType[float]:
        loss = sequence_cross_entropy_loss(
            result.output_logits_s, result.input, self._length, self._n_values
        )
        return self._weight * loss


class ReceiverMessageCrossEntropyLoss(nn.Module):
    def __init__(self, length: int, n_values: int, weight: float = 1.0):
        super().__init__()
        self._length = length
        self._n_values = n_values
        self._weight = weight

    def forward(self, result: GameResult) -> TensorType[float]:
        loss = [
            sequence_cross_entropy_loss(
                logits, result.message_s, self._length, self._n_values
            )
            for logits in result.message_logits_r  # type: ignore
        ]
        loss_sum = torch.stack(loss, dim=-1).sum(dim=-1)
        return self._weight * loss_sum


class SenderAutoEncodingCrossEntropyLoss(nn.Module):
    def __init__(self, length: int, n_values: int, weight: float = 1.0):
        super().__init__()
        self._length = length
        self._n_values = n_values
        self._weight = weight

    def forward(self, result: GameResult) -> TensorType[float]:
        loss = sequence_cross_entropy_loss(
            result.output_logits_auto_encoding_s,
            result.input,
            self._length,
            self._n_values,
        )
        return self._weight * loss


class SenderAutoEncodingLeaveCrossEntropyLoss(nn.Module):
    def __init__(self, length: int, n_values: int, weight: float = 1.0):
        super().__init__()
        self._length = length
        self._n_values = n_values
        self._weight = weight

    def forward(self, result: GameResult) -> TensorType[float]:
        mask = (result.output_auto_encoding_s != result.input).any(dim=-1).float()
        loss = -mask * sequence_cross_entropy_loss(
            result.message_logits_s,
            result.message_s,
            self._length,
            self._n_values,
        )
        return self._weight * loss


class ReceiverAutoEncodingCrossEntropyLoss(nn.Module):
    def __init__(self, length: int, n_values: int, weight: float = 1.0):
        super().__init__()
        self._length = length
        self._n_values = n_values
        self._weight = weight

    def forward(self, result: GameResult) -> TensorType[float]:
        loss = [
            sequence_cross_entropy_loss(
                logits, result.message_s, self._length, self._n_values
            )
            for logits in result.message_logits_auto_encoding_r  # type: ignore
        ]
        loss_sum = torch.stack(loss, dim=-1).sum(dim=-1)
        return self._weight * loss_sum


class Loss(nn.Module):
    def __init__(
        self,
        object_length: int,
        object_n_values: int,
        message_length: int,
        message_n_values: int,
        sender_loss: nn.Module,
        receiver_loss: nn.Module,
        additional_losses: Iterable[nn.Module] | None = None,
    ):
        super().__init__()
        self._object_length = object_length
        self._object_n_values = object_n_values
        self._message_length = message_length
        self._message_n_values = message_n_values
        self._sender_loss = sender_loss
        self._receiver_loss = receiver_loss
        self._additional_losses = nn.ModuleList(additional_losses or [])

    def forward(self, result: GameResult) -> TensorType[float]:
        loss_r = self._receiver_loss(result)
        loss_s = self._sender_loss(result, -loss_r)
        total_loss = loss_r + loss_s

        for loss in self._additional_losses:
            total_loss += loss(result)

        return total_loss.mean()
