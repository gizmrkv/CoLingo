import copy
import torch as th
from typing import Any

from . import baseline


def build_optimizer_type(optimizer_type: str):
    optimizers_dict = {"adam": th.optim.Adam}
    return optimizers_dict[optimizer_type]


class Agent(th.nn.Module):
    def __init__(
        self,
        model: th.nn.Module,
        baseline: baseline.Baseline,
        optimizer: str,
        optimizer_params: dict,
        name: str | None = None,
    ):
        super().__init__()
        self.model = copy.deepcopy(model)
        for param in self.model.parameters():
            th.nn.init.normal_(param)

        self.baseline = copy.deepcopy(baseline)
        self.optimizer = build_optimizer_type(optimizer)(
            self.model.parameters(), **optimizer_params
        )
        self.name = name

    def forward(self, x: th.Tensor, input_type: str):
        return self.model(x, input_type)

    def loss(self, reward: th.Tensor, x: th.Tensor, aux: Any):
        self.baseline.update(x, reward)
        return -aux.log_prob.mean() * (
            reward - self.baseline(x) - 0.0 * aux.entropy.mean()
        )
