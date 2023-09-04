import math
from typing import Callable, Dict, Iterable, Mapping

import torch
import torch.nn.functional as F
from torchtyping import TensorType

from ...core import Computable, Loggable
from ...module.reinforce_loss import ReinforceLoss
from .game import RecoNetworkAgent, RecoNetworkGameResult, RecoNetworkSubGameResult


class Loss(
    Computable[TensorType[..., int], RecoNetworkGameResult, TensorType[1, float]]
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
        receiver_imitation_loss_weight: float = 0.0,
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
        self.receiver_imitation_loss_weight = receiver_imitation_loss_weight

        self.reinforce_loss = ReinforceLoss(
            max_len=message_max_len,
            entropy_weight=entropy_weight,
            length_weight=length_weight,
            baseline=baseline,
            length_baseline=baseline,
        )

    def compute(
        self,
        input: TensorType[..., int],
        output: RecoNetworkGameResult,
        step: int | None = None,
    ) -> TensorType[1, float]:
        return self.total_loss(output).mean()

    def receivers_loss(
        self, result: RecoNetworkSubGameResult
    ) -> Dict[str, TensorType[..., float]]:
        return {
            name: F.cross_entropy(
                logits.view(-1, self.concept_values),
                result.input.view(-1),
                reduction="none",
            )
            .view(-1, self.concept_length)
            .mean(dim=-1)
            for name, logits in result.outputs_logits.items()
        }

    def sender_loss(
        self, result: RecoNetworkSubGameResult, receiver_loss: TensorType[..., float]
    ) -> TensorType[..., float]:
        return self.reinforce_loss(
            reward=-receiver_loss.detach(),
            log_prob=result.message_log_prob,
            entropy=result.message_entropy,
            length=result.message_length,
        )

    def sub_game_loss(self, result: RecoNetworkSubGameResult) -> TensorType[..., float]:
        if len(result.receivers) == 0:
            # Return dummy loss if there is no receiver
            return torch.zeros(
                result.input.size(0), device=result.input.device, dtype=torch.float32
            )

        loss_rs = self.receivers_loss(result)
        loss_r = torch.stack(list(loss_rs.values()), dim=-1).mean(dim=-1)
        loss_s = self.sender_loss(result, loss_r)
        loss = self.receiver_loss_weight * loss_r + self.sender_loss_weight * loss_s

        if not math.isclose(self.receiver_imitation_loss_weight, 0.0):
            loss_ris = self.receiver_imitation_loss(result)
            loss_ri = torch.stack(list(loss_ris.values()), dim=-1).mean(dim=-1)
            loss += self.receiver_imitation_loss_weight * loss_ri

        return loss

    def total_loss(self, output: RecoNetworkGameResult) -> TensorType[..., float]:
        return torch.stack(
            [self.sub_game_loss(result) for result in output.sub_results.values()],
            dim=-1,
        ).mean(dim=-1)

    def receiver_imitation_loss(
        self, result: RecoNetworkSubGameResult
    ) -> Dict[str, TensorType[..., float]]:
        losses = {}
        for name, receiver in result.receivers.items():
            correct_mask = (result.outputs[name] == result.input).all(dim=-1).float()
            feature = receiver.concept_encoder(result.outputs[name])
            _, message_logits = receiver.message_decoder(
                feature, message=result.message
            )
            loss = correct_mask * F.cross_entropy(
                message_logits.view(-1, self.message_vocab_size),
                result.message.view(-1),
                reduction="none",
            ).view(-1, self.message_max_len).mean(dim=-1)
            losses[name] = loss

        return losses
