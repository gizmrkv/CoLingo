import random
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Iterable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ..core import Callback
from ..logger import Logger


@dataclass
class InferringGameResult:
    agent: nn.Module
    input: torch.Tensor
    output: torch.Tensor
    info: Any


class InferringGame(nn.Module):
    def __init__(
        self,
        agent: nn.Module,
        input_command: str = "input",
        output_command: str = "output",
    ):
        super().__init__()
        self._agent = agent
        self._input_command = input_command
        self._output_command = output_command

    def forward(self, input: torch.Tensor) -> InferringGameResult:
        latent = self._agent(input=input, command=self._input_command)
        output, info = self._agent(latent=latent, command=self._output_command)
        result = InferringGameResult(self._agent, input, output, info)
        return result
