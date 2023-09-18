import math
from typing import Callable

from torch import nn
from torchtyping import TensorType


class ReinforceLoss(nn.Module):
    def __init__(
        self,
        baseline: Callable[[TensorType[..., float]], TensorType[..., float]]
        | None = None,
    ) -> None:
        super().__init__()
        self.baseline = baseline

    def forward(
        self,
        reward: TensorType[..., float],
        log_prob: TensorType[..., float],
    ) -> TensorType[..., float]:
        reward = reward.detach()
        log_prob = log_prob.sum(dim=-1)

        if self.baseline is not None:
            reward -= self.baseline(reward)
        loss = -reward * log_prob

        # if entropy is not None and not math.isclose(self.entropy_weight, 0.0):
        #     ent_loss = self.entropy_weight * entropy.sum(dim=-1)
        #     loss -= ent_loss

        return loss
