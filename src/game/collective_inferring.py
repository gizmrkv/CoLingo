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
class CollectiveInferringGameResult:
    agents: dict[str, Any]
    input: Any
    outputs: dict[str, Any]
    infos: dict[str, Any]


class CollectiveInferringGame(nn.Module):
    def __init__(self, input_command: str = "input", output_command: str = "output"):
        super().__init__()
        self.input_command = input_command
        self.output_command = output_command

    def forward(self, agents: dict, input) -> CollectiveInferringGameResult:
        outputs = {}
        infos = {}
        for agent_name in agents:
            agent = agents[agent_name]
            latent = agent(input=input, command=self.input_command)
            output, info = agent(latent=latent, command=self.output_command)
            outputs[agent_name] = output
            infos[agent_name] = info
        result = CollectiveInferringGameResult(
            agents=agents, input=input, outputs=outputs, infos=infos
        )
        return result


class CollectiveInferringGameTrainer(Callback):
    pass


class CollectiveInferringGameEvaluator(Callback):
    def __init__(
        self,
        game: CollectiveInferringGame,
        agents: dict[str, nn.Module],
        input: torch.Tensor,
        metric: Callable[[CollectiveInferringGameResult], dict],
        logger: Logger | Iterable[Logger],
        name: str,
        run_on_begin: bool = True,
        run_on_end: bool = True,
    ):
        super().__init__()
        self.game = game
        self.agents = agents
        self.input = input
        self.metric = metric
        self.loggers = [logger] if isinstance(logger, Logger) else logger
        self.name = name
        self.run_on_begin = run_on_begin
        self.run_on_end = run_on_end

    def on_begin(self):
        if self.run_on_begin:
            self.evaluate()

    def on_end(self):
        if self.run_on_end:
            self.evaluate()

    def on_update(self, iteration: int):
        self.evaluate()

    def evaluate(self):
        result = self.game(self.agents, self.input)
        metrics = self.metric(result)
        for logger in self.loggers:
            logger.log({self.name: metrics})
