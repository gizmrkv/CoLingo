import random
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Iterable, Tuple

import torch as th
from torch.utils.data import DataLoader

from ..core import Callback
from ..logger import Logger


@dataclass
class InferringGameResult:
    input: th.Tensor
    output: th.Tensor
    target: th.Tensor
    loss: th.Tensor | None = None


class InferringGame(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        agent: th.nn.Module,
        input: th.Tensor,
        target: th.Tensor,
        loss: th.nn.Module | None = None,
        input_command: str = "input",
        output_command: str = "output",
    ):
        hidden = agent(input=input, command=input_command)
        output = agent(hidden=hidden, command=output_command)
        result = InferringGameResult(input=input, output=output, target=target)
        if self.training:
            result.loss = loss(output, target)

        return result


class InferringGameTrainer(Callback):
    def __init__(
        self,
        agents: dict[str, th.nn.Module],
        optimizers: dict[str, th.optim.Optimizer],
        dataloader: DataLoader,
        loss: th.nn.Module,
        max_batches: int = 1,
        input_command: str = "input",
        output_command: str = "output",
    ):
        super().__init__()
        self.agents = agents
        self.optimizers = optimizers
        self.dataloader = dataloader
        self.loss = loss
        self.max_batches = max_batches
        self.input_command = input_command
        self.output_command = output_command

        self.game = InferringGame()
        self.game.train()

        self.agent_names = list(agents.keys())

    def on_update(self, iteration: int):
        agent_name = random.choice(self.agent_names)
        agent = self.agents[agent_name]
        optimizer = self.optimizers[agent_name]

        agent.train()
        optimizer.zero_grad()

        for input, target in islice(self.dataloader, self.max_batches):
            result: InferringGameResult = self.game(
                agent=agent,
                input=input,
                target=target,
                loss=self.loss,
                input_command=self.input_command,
                output_command=self.output_command,
            )
            result.loss.sum().backward(retain_graph=True)
            optimizer.step()


class InferringGameEvaluator(Callback):
    def __init__(
        self,
        agents: dict[str, th.nn.Module],
        input: th.Tensor,
        target: th.Tensor,
        metric: Callable[[InferringGameResult], dict],
        logger: Logger | Iterable[Logger],
        name: str,
        run_on_begin: bool = True,
        run_on_end: bool = True,
        input_command: str = "input",
        output_command: str = "output",
    ):
        super().__init__()
        self.agents = agents
        self.input = input
        self.target = target
        self.metric = metric
        self.loggers = [logger] if isinstance(logger, Logger) else logger
        self.name = name
        self.run_on_begin = run_on_begin
        self.run_on_end = run_on_end
        self.input_command = input_command
        self.output_command = output_command

        self.game = InferringGame()
        self.game.eval()

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

            with th.no_grad():
                result: InferringGameResult = self.game(
                    agent=agent,
                    input=self.input,
                    target=self.target,
                    input_command=self.input_command,
                    output_command=self.output_command,
                )

            metric = self.metric(result)
            log |= {agent_name: metric}

        for logger in self.loggers:
            logger.log({self.name: log})
