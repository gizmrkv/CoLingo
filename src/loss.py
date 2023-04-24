import torch as th
from typing import Any

from . import baseline


class ReinforceLoss(th.nn.Module):
    def __init__(self, baseline: baseline.Baseline, entropy_weight: float = 0.0):
        super().__init__()
        self.baseline = baseline
        self.entropy_weight = entropy_weight

    def forward(self, reward: th.Tensor, x: th.Tensor, aux: Any):
        self.baseline.update(x, reward)
        return -aux.log_prob.mean() * (
            reward - self.baseline(x) - self.entropy_weight * aux.entropy.mean()
        )
