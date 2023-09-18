import math
from typing import Callable, Dict, Literal

import torch
import torch.nn.functional as F
from torchtyping import TensorType

from ...module.reinforce_loss import ReinforceLoss
from .game import RecoNetworkGameResult, RecoNetworkSubGameResult


class Loss:
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
        length_accuracy_mask: Literal["complete", "partial", "none"] = "complete",
        receiver_loss_weight: float = 1.0,
        sender_loss_weight: float = 1.0,
        sender_imitation_loss_weight: float = 0.0,
        sender_imitation_mask: Literal["complete", "partial", "none"] = "complete",
        receiver_imitation_loss_weight: float = 0.0,
        receiver_imitation_mask: Literal["complete", "partial", "none"] = "complete",
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
        self.length_accuracy_mask = length_accuracy_mask
        self.receiver_loss_weight = receiver_loss_weight
        self.sender_loss_weight = sender_loss_weight
        self.sender_imitation_loss_weight = sender_imitation_loss_weight
        self.receiver_imitation_loss_weight = receiver_imitation_loss_weight
        self.sender_imitation_mask = sender_imitation_mask
        self.receiver_imitation_mask = receiver_imitation_mask

        self.reinforce_loss = ReinforceLoss(baseline)
        self.reinforce_loss_length = ReinforceLoss(length_baseline)

    def __call__(self, output: RecoNetworkGameResult) -> TensorType[1, float]:
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
        loss = self.reinforce_loss(
            reward=-receiver_loss.detach(),
            log_prob=result.message_log_prob,
        )

        if not math.isclose(self.length_weight, 0.0):
            loss -= self.entropy_weight * result.message_entropy.sum(dim=-1)

        if not math.isclose(self.length_weight, 0.0):
            acc_masks = []
            for output in result.outputs.values():
                acc_mask = self.accuracy_mask(
                    output, result.input, self.length_accuracy_mask
                )
                acc_masks.append(acc_mask)

            acc_mask = torch.stack(acc_masks, dim=-1).mean(dim=-1)

            loss += (
                self.length_weight
                * acc_mask
                * self.reinforce_loss_length(
                    reward=-result.message_length / self.message_max_len,
                    log_prob=result.message_log_prob,
                )
            )

        return loss

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

        if not math.isclose(self.sender_imitation_loss_weight, 0.0):
            loss_sis = self.sender_imitation_loss(result)
            loss_si = torch.stack(list(loss_sis.values()), dim=-1).mean(dim=-1)
            loss += self.sender_imitation_loss_weight * loss_si

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

    def sender_imitation_loss(
        self, result: RecoNetworkSubGameResult
    ) -> Dict[str, TensorType[..., float]]:
        losses = {}
        for name, receiver in result.receivers.items():
            feature = receiver.concept_encoder(result.outputs[name])
            _, message_logits = receiver.message_decoder(
                feature, message=result.message
            )
            # Should I mask the imitation loss?
            # loss = result.message_mask * F.cross_entropy(
            loss = (
                F.cross_entropy(
                    message_logits.view(-1, self.message_vocab_size),
                    result.message.view(-1),
                    reduction="none",
                )
                .view(-1, self.message_max_len)
                .mean(dim=-1)
            )
            acc_mask = self.accuracy_mask(
                result.outputs[name], result.input, self.sender_imitation_mask
            )
            loss = acc_mask * loss
            losses[name] = loss

        return losses

    def receiver_imitation_loss(
        self, result: RecoNetworkSubGameResult
    ) -> Dict[str, TensorType[..., float]]:
        losses = {}
        feature = result.sender.message_encoder(result.message)
        _, concept_logits = result.sender.concept_decoder(
            feature, message=result.message
        )
        for name_r, output in result.outputs.items():
            acc_mask = self.accuracy_mask(
                output, result.input, self.receiver_imitation_mask
            )
            loss = acc_mask * F.cross_entropy(
                concept_logits.view(-1, self.concept_values),
                output.view(-1),
                reduction="none",
            ).view(-1, self.concept_length).mean(dim=-1)
            losses[name_r] = loss

        return losses

    def accuracy_mask(
        self,
        input: TensorType[..., int],
        target: TensorType[..., int],
        mask_mode: Literal["complete", "partial", "none"],
    ) -> TensorType[..., float]:
        if mask_mode == "complete":
            return (input == target).all(dim=-1).float()
        elif mask_mode == "partial":
            return (input == target).float().mean(dim=-1)
        elif mask_mode == "none":
            return torch.ones_like(input)
