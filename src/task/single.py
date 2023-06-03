import random
from itertools import islice
from typing import Iterable

import torch as th
from torch.utils.data import DataLoader

from ..core.agent import Agent
from ..core.callback import Callback
from ..core.logger import Logger
from ..core.metric import Metric


class SingleTrainer(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        optimizers: dict[str, th.optim.Optimizer],
        dataloader: DataLoader,
        loss: th.nn.Module,
        input_key,
        output_key,
        max_batches: int = 1,
    ):
        super().__init__()

        self.agents = agents
        self.optimizers = optimizers
        self.dataloader = dataloader
        self.loss = loss
        self.input_key = input_key
        self.output_key = output_key
        self.max_batches = max_batches

        self.agent_names = list(self.agents.keys())

    def on_update(self):
        for input, target in islice(self.dataloader, self.max_batches):
            agent_name = random.choice(self.agent_names)
            agent = self.agents[agent_name]
            optimizer = self.optimizers[agent_name]

            agent.train()
            optimizer.zero_grad()

            hidden = agent.input({self.input_key: input})
            (output,) = agent.output(self.output_key, hidden=hidden)

            loss: th.Tensor = self.loss(input=output, target=target).mean()
            loss.backward(retain_graph=True)
            optimizer.step()


class SingleEvaluator(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        input: th.Tensor,
        target: th.Tensor,
        metrics: Iterable[Metric],
        loggers: Iterable[Logger],
        input_key,
        output_key,
        name: str,
        interval: int = 1,
    ):
        super().__init__()
        self.agents = agents
        self.input = input
        self.target = target
        self.metrics = metrics
        self.loggers = loggers
        self.input_key = input_key
        self.output_key = output_key
        self.interval = interval
        self.name = name

        self.agent_names = list(self.agents.keys())

        self._count = 0

    def on_update(self):
        if self._count % self.interval != 0:
            return

        self._count += 1

        logs = {agent_name: {} for agent_name in self.agent_names}
        for agent_name in self.agent_names:
            agent = self.agents[agent_name]
            agent.eval()
            with th.no_grad():
                hidden = agent.input({self.input_key: self.input})
                (output,) = agent.output(self.output_key, hidden=hidden)

            for metric in self.metrics:
                met = metric.calculate(
                    input=self.input, output=output, target=self.target
                )
                logs[agent_name] |= met

        for logger in self.loggers:
            logger.log({self.name: logs})
