from typing import Callable

from torch import nn
from torchtyping import TensorType


class ReinforceLoss(nn.Module):
    def __init__(
        self,
        max_len: int,
        entropy_weight: float = 0.0,
        length_weight: float = 0.0,
        baseline: Callable[[TensorType[..., float]], TensorType[..., float]]
        | None = None,
        length_baseline: Callable[[TensorType[..., float]], TensorType[..., float]]
        | None = None,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.entropy_weight = entropy_weight
        self.length_weight = length_weight
        self.baseline = baseline
        self.length_baseline = length_baseline

    def forward(
        self,
        reward: TensorType[..., float],
        log_prob: TensorType[..., "max_len", float],
        entropy: TensorType[..., "max_len", float] | None = None,
        length: TensorType[..., int] | None = None,
    ) -> TensorType[..., float]:
        reward = reward.detach()
        log_prob = log_prob.sum(dim=-1)

        if self.baseline is not None:
            reward -= self.baseline(reward)
        loss = -reward * log_prob

        if entropy is not None:
            ent_loss = self.entropy_weight * entropy.sum(dim=-1)
            loss -= ent_loss

        if length is not None:
            len_loss = self.length_weight * length.float() / self.max_len
            if self.length_baseline is not None:
                len_loss -= self.length_baseline(len_loss)
            loss += len_loss * log_prob

        return loss
