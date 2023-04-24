import copy
import torch as th
from typing import Any

from . import baseline


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
        self.optimizer = {
            "adam": th.optim.Adam,
            "sgd": th.optim.SGD,
            "adagrad": th.optim.Adagrad,
            "adadelta": th.optim.Adadelta,
            "rmsprop": th.optim.RMSprop,
            "sparseadam": th.optim.SparseAdam,
            "adamax": th.optim.Adamax,
            "asgd": th.optim.ASGD,
            "lbfgs": th.optim.LBFGS,
            "rprop": th.optim.Rprop,
        }[optimizer.lower()](self.model.parameters(), **optimizer_params)
        self.name = name

        for m in self.model.modules():
            if isinstance(m, th.nn.Linear):
                th.nn.init.normal_(m.weight)
                th.nn.init.zeros_(m.bias)

    def forward(self, x: th.Tensor, input_type: str):
        return self.model(x, input_type)
