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
        loss: th.nn.Module,
        optimizer: str,
        optimizer_params: dict,
        name: str | None = None,
    ):
        super().__init__()
        self.model = model 
        self.loss = loss
        self.optimizer = build_optimizer_type(optimizer)(
            self.model.parameters(), **optimizer_params
        )
        self.name = name

        for m in self.model.modules():
            if isinstance(m, th.nn.Linear):
                print(m)
                th.nn.init.normal_(m.weight)
                th.nn.init.zeros_(m.bias)


    def forward(self, x: th.Tensor, input_type: str):
        return self.model(x, input_type)
