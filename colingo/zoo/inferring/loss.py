from typing import Literal

import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torchtyping import TensorType

from ...baseline import BatchMeanBaseline
from ...loss import ReinforceLoss
from .game import GameResult


class Loss(nn.Module):
    def __init__(
        self,
        n_values: int,
        use_reinforce: bool,
        baseline: Literal["batch_mean"] = "batch_mean",
        entropy_weight: float = 0.0,
    ):
        super().__init__()

        self._n_values = n_values
        self._use_reinforce = use_reinforce
        if use_reinforce:
            self._entropy_weight = entropy_weight
            baselines = {"batch_mean": BatchMeanBaseline}
            self._baseline = baselines[baseline]()
            self._reinforce_loss = ReinforceLoss(
                entropy_weight=entropy_weight, baseline=self._baseline
            )

    def forward(self, result: GameResult) -> TensorType[float]:
        if self._use_reinforce:
            acc = (result.output == result.input).float().mean(dim=-1)
            distr = Categorical(logits=result.logits)
            log_prob = distr.log_prob(result.output)
            entropy = distr.entropy()
            return self._reinforce_loss(acc, log_prob, entropy)
        else:
            return F.cross_entropy(
                result.logits.view(-1, self._n_values), result.input.view(-1)
            )
