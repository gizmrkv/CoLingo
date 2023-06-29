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
    agent: Any
    input: Any
    output: Any
    info: Any


class InferringGame(nn.Module):
    def __init__(self, input_command: str = "input", output_command: str = "output"):
        super().__init__()
        self.input_command = input_command
        self.output_command = output_command

    def forward(self, agent, input) -> InferringGameResult:
        latent = agent(input=input, command=self.input_command)
        output, info = agent(latent=latent, command=self.output_command)
        result = InferringGameResult(agent=agent, input=input, output=output, info=info)
        return result


class InferringGameTrainer(Callback):
    def __init__(
        self,
        game: InferringGame,
        agents: dict[str, nn.Module],
        optimizers: dict[str, optim.Optimizer],
        dataloader: DataLoader,
        loss: Callable[[InferringGameResult, torch.Tensor], torch.Tensor],
        max_batches: int = 1,
    ):
        super().__init__()
        self.game = game
        self.agents = agents
        self.optimizers = optimizers
        self.dataloader = dataloader
        self.loss = loss
        self.max_batches = max_batches

        self.agent_names = list(agents.keys())

    def on_update(self, iteration: int):
        agent_name = random.choice(self.agent_names)
        agent = self.agents[agent_name]
        optimizer = self.optimizers.get(agent_name)

        if optimizer is None:
            return

        agent.train()
        optimizer.zero_grad()

        for input, target in islice(self.dataloader, self.max_batches):
            result = self.game(agent=agent, input=input)
            loss: torch.Tensor = self.loss(result, target)
            loss.sum().backward(retain_graph=True)
            optimizer.step()


class InferringGameEvaluator(Callback):
    def __init__(
        self,
        game: InferringGame,
        agents: dict[str, nn.Module],
        input: torch.Tensor,
        metric: Callable[[InferringGameResult], dict],
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
        log = {}
        for agent_name, agent in self.agents.items():
            agent.eval()

            with torch.no_grad():
                result = self.game(agent=agent, input=self.input)

            metric = self.metric(result)
            log |= {agent_name: metric}

        for logger in self.loggers:
            logger.log({self.name: log})
