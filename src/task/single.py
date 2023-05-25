import random
from itertools import islice
from typing import Callable, Iterable

import torch as th
from networkx import DiGraph
from torch.utils.data import DataLoader

from ..core.agent import Agent
from ..core.callback import Callback
from ..core.command import Command
from ..core.logger import Logger
from ..core.network import generate_custom_graph


class SingleTrainer(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        dataloader: DataLoader,
        loss: th.nn.Module,
        max_batches: int = 1,
        network: DiGraph | None = None,
        command: Command = Command.PREDICT,
    ):
        super().__init__()

        self.agents = agents
        self.network = network
        self.dataloader = dataloader
        self.loss = loss
        self.command = command
        self.max_batches = max_batches

        if self.network is None:
            self.network = generate_custom_graph(list(self.agents.keys()))

        self._nodes = list(self.network.nodes)

    def on_update(self):
        for input, target in islice(self.dataloader, self.max_batches):
            agent = self.agents[random.choice(self._nodes)]
            agent.train()
            output = agent(input, self.command)
            loss = self.loss(input=output, target=target).mean()
            agent.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            agent.step()


class SingleEvaluator(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        dataloader: DataLoader,
        metrics: dict[str, Callable],
        loggers: Iterable[Logger],
        name: str,
        interval: int = 1,
        network: DiGraph | None = None,
        command: Command = Command.PREDICT,
    ):
        super().__init__()
        self.agents = agents
        self.dataloader = dataloader
        self.metrics = metrics
        self.loggers = loggers
        self.name = name
        self.interval = interval
        self.network = network
        self.command = command

        if self.network is None:
            self.network = generate_custom_graph(list(self.agents.keys()))

        self._nodes = list(self.network.nodes)
        self._count = 0

    def on_update(self):
        if self._count % self.interval != 0:
            return

        self._count += 1

        for input, target in self.dataloader:
            logs = {agent_name: {} for agent_name in self._nodes}
            for agent_name in self._nodes:
                agent = self.agents[agent_name]
                agent.eval()
                with th.no_grad():
                    output = agent(input, self.command)

                for metric_name, metric in self.metrics.items():
                    value = metric(input=input, output=output, target=target)
                    logs[agent_name][metric_name] = value

            for logger in self.loggers:
                logger.log({self.name: logs})

            break
