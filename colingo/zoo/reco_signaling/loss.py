from typing import Callable

import torch.nn.functional as F
from torchtyping import TensorType

from ...core import Computable
from ...module.reinforce_loss import ReinforceLoss
from .game import RecoSignalingGameResult


class Loss(
    Computable[TensorType[..., int], RecoSignalingGameResult, TensorType[1, float]]
):
    def __init__(
        self,
        concept_length: int,
        concept_values: int,
        message_max_len: int,
        message_vocab_size: int,
        entropy_weight: float = 0.0,
        length_weight: float = 0.0,
        baseline: Callable[[TensorType[..., float]], TensorType[..., float]]
        | None = None,
        length_baseline: Callable[[TensorType[..., float]], TensorType[..., float]]
        | None = None,
        receiver_loss_weight: float = 1.0,
        sender_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.concept_length = concept_length
        self.concept_values = concept_values
        self.message_max_len = message_max_len
        self.message_vocab_size = message_vocab_size
        self.entropy_weight = entropy_weight
        self.length_weight = length_weight
        self.baseline = baseline
        self.length_baseline = length_baseline
        self.receiver_loss_weight = receiver_loss_weight
        self.sender_loss_weight = sender_loss_weight

        self.reinforce_loss = ReinforceLoss(
            max_len=message_max_len,
            entropy_weight=entropy_weight,
            length_weight=length_weight,
            baseline=baseline,
            length_baseline=baseline,
        )

    def receiver_loss(self, result: RecoSignalingGameResult) -> TensorType[..., float]:
        return (
            F.cross_entropy(
                result.output_logits.view(-1, self.concept_values),
                result.input.view(-1),
                reduction="none",
            )
            .view(-1, self.concept_length)
            .mean(dim=-1)
        )

    def sender_loss(
        self, result: RecoSignalingGameResult, receiver_loss: TensorType[..., float]
    ) -> TensorType[..., float]:
        return self.reinforce_loss(
            reward=-receiver_loss.detach(),
            log_prob=result.message_log_prob,
            entropy=result.message_entropy,
            length=result.message_length,
        )

    def total_loss(self, result: RecoSignalingGameResult) -> TensorType[..., float]:
        loss_r = self.receiver_loss(result)
        loss_s = self.sender_loss(result, loss_r)
        return self.receiver_loss_weight * loss_r + self.sender_loss_weight * loss_s

    def compute(
        self,
        input: TensorType[..., int],
        output: RecoSignalingGameResult,
        step: int | None = None,
    ) -> TensorType[1, float]:
        loss_r = self.receiver_loss(output)
        loss_s = self.sender_loss(output, loss_r)
        loss_total = loss_r + loss_s
        return loss_total.mean()
