from dataclasses import dataclass
from typing import Any

from torch import nn
from torchtyping import TensorType

BATCH = "batch"
LENGTH = "length"
N_VALUES = "n_values"


@dataclass
class GameResult:
    agent: nn.Module
    input: TensorType[BATCH, LENGTH, int]
    latent: Any
    output: TensorType[BATCH, LENGTH, int]
    logits: TensorType[BATCH, LENGTH, N_VALUES, float]


class Game(nn.Module):
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

    def forward(self, input: TensorType[BATCH, LENGTH]) -> GameResult:
        latent = self._agent(object=input, command=self._input_command)
        output, logits = self._agent(latent=latent, command=self._output_command)
        return GameResult(self._agent, input, latent, output, logits)
