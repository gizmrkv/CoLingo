import copy
import torch as th
from typing import Any

from . import baseline


class Agent(th.nn.Module):
    def __init__(
        self,
        model: th.nn.Module,
        optimizer: th.optim.Optimizer,
        optimizer_params: dict[str, Any],
        name: str,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
        self.name = name

        for m in self.model.modules():
            if isinstance(m, th.nn.Linear):
                th.nn.init.kaiming_uniform_(m.weight)
                th.nn.init.zeros_(m.bias)

    def forward(self, x: th.Tensor, input_type: str):
        return self.model(x, input_type)
