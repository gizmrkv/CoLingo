import copy
import torch as th
from typing import Any

from . import baseline


class Agent(th.nn.Module):
    def __init__(
        self,
        model: th.nn.Module,
        tasks: dict[str, dict],
        name: str,
    ):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.name = name

        for task_name, task_params in tasks.items():
            if "optimizer" in task_params.keys():
                self.tasks[task_name]["optimizer"] = task_params["optimizer"](
                    self.model.parameters(), **task_params["optimizer_params"]
                )

        for m in self.model.modules():
            if isinstance(m, th.nn.Linear):
                th.nn.init.normal_(m.weight)
                th.nn.init.zeros_(m.bias)

    def forward(self, x: th.Tensor, input_type: str):
        return self.model(x, input_type)
