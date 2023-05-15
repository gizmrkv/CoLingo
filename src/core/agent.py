import random

import torch as th


class Agent(th.nn.Module):
    def __init__(
        self,
        model: th.nn.Module,
        optimizer: th.optim.Optimizer,
        optimizer_params: dict,
        step_prob:float = 1.0,
        name: str | None = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
        self.step_prob = step_prob
        self.name = name

        for m in self.model.modules():
            if isinstance(m, th.nn.Linear):
                th.nn.init.kaiming_uniform_(m.weight)
                th.nn.init.zeros_(m.bias)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step(self):
        if random.random() < self.step_prob:
            self.optimizer.step()
