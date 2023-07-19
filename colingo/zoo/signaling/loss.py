from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torchtyping import TensorType

from ...baseline import BatchMeanBaseline
from ...loss import ReinforceLoss
from .game import GameResult

BATCH = "batch"
OBJECT_LENGTH = "object_length"
OBJECT_N_VALUES = "object_n_values"


def object_loss(
    input: TensorType[BATCH, OBJECT_LENGTH, OBJECT_N_VALUES, float],
    target: TensorType[BATCH, OBJECT_LENGTH, int],
    object_length: int,
    object_n_values: int,
) -> TensorType[BATCH, float]:
    input = input.view(-1, object_n_values)
    target = target.view(-1)
    loss = F.cross_entropy(input, target, reduction="none")
    return loss.view(-1, object_length).sum(dim=-1)


def object_reinforce_loss(
    output: TensorType[BATCH, OBJECT_LENGTH, int],
    target: TensorType[BATCH, OBJECT_LENGTH, int],
    logits: TensorType[BATCH, OBJECT_LENGTH, OBJECT_N_VALUES, float],
    reinforce_loss: ReinforceLoss,
) -> TensorType[BATCH, float]:
    acc = (output == target).float().mean(dim=-1)
    distr = Categorical(logits=logits)
    log_prob = distr.log_prob(output)
    entropy = distr.entropy()
    return reinforce_loss(acc, log_prob, entropy)


class Loss(nn.Module):
    def __init__(
        self,
        object_length: int,
        object_n_values: int,
        message_length: int,
        message_n_values: int,
        use_reinforce: bool,
        baseline: Literal["batch_mean"] = "batch_mean",
        entropy_weight: float = 0.0,
    ):
        super().__init__()

        self._object_length = object_length
        self._object_n_values = object_n_values
        self._message_length = message_length
        self._message_n_values = message_n_values
        self._use_reinforce = use_reinforce
        self._entropy_weight = entropy_weight
        baselines = {"batch_mean": BatchMeanBaseline}
        self._baseline = baselines[baseline]()
        self._reinforce_loss = ReinforceLoss(
            entropy_weight=entropy_weight, baseline=self._baseline
        )

    def forward(self, result: GameResult) -> TensorType[float]:
        output_loss_r = [
            object_reinforce_loss(output, result.input, logits, self._reinforce_loss)
            if self._use_reinforce
            else object_loss(
                logits, result.input, self._object_length, self._object_n_values
            )
            for output, logits in zip(result.output_r, result.output_logits_r)
        ]
        output_loss_r_mean = torch.stack(output_loss_r, dim=-1).sum(dim=-1)

        message_loss_s = self._reinforce_loss(
            -output_loss_r_mean.detach(),
            result.message_log_prob_s,
            result.message_entropy_s,
            result.message_length_s,
        )

        total_loss = output_loss_r_mean + message_loss_s

        # ltt = result.sender(message=result.message_s, command=result.receive_command)
        # _, logits = result.sender(latent=ltt, command=result.output_command)
        # loss = object_loss(
        #     logits, result.input, self._object_length, self._object_n_values
        # )
        # total_loss += loss

        for receiver, output in zip(result.receivers, result.output_r):
            ltt = receiver(object=output, command=result.input_command)
            _, logits = receiver(latent=ltt, command=result.send_command)
            loss = object_loss(
                logits, result.message_s, self._message_length, self._message_n_values
            )
            total_loss += loss * 0.01

        return total_loss.sum()
