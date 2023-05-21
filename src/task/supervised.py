import random
from itertools import islice
from typing import Callable

import torch as th
from networkx import DiGraph
from torch.utils.data import DataLoader

from ..core.agent import Agent
from ..core.callback import Callback
from ..core.command import Command
from ..core.logger import Logger
from ..core.network import generate_custom_graph


class SupervisedTrainer(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        dataloader: DataLoader,
        loss: th.nn.Module,
        network: DiGraph | None = None,
        max_batches: int = 1,
        command: Command = Command.PREDICT,
        name: str | None = None,
    ):
        super().__init__()

        self.agents = agents
        self.network = network
        self.dataloader = dataloader
        self.loss = loss
        self.command = command
        self.max_batches = max_batches
        self.name = name

        if self.network is None:
            self.network = generate_custom_graph(list(self.agents.keys()))

        self._nodes = list(self.network.nodes)

    def on_update(self):
        for batch, target in islice(self.dataloader, self.max_batches):
            agent = self.agents[random.choice(self._nodes)]
            agent.train()
            output = agent(batch, self.command)
            loss = self.loss(output, target).mean()
            agent.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            agent.step()


class SupervisedEvaluator(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        dataloader: DataLoader,
        metrics: dict[str, Callable],
        loggers: dict[str, Logger],
        network: DiGraph | None = None,
        command: Command = Command.PREDICT,
        interval: int = 10,
        name: str = "eval",
    ):
        super().__init__()
        self.agents = agents
        self.network = network
        self.dataloader = dataloader
        self.metrics = metrics
        self.loggers = loggers
        self.command = command
        self.interval = interval
        self.name = name

        if self.network is None:
            self.network = generate_custom_graph(list(self.agents.keys()))

        self._nodes = list(self.network.nodes)
        self._count = 0

    def on_update(self):
        if self._count % self.interval != 0:
            return

        self._count += 1

        for batch, target in self.dataloader:
            logs = {agent_name: {} for agent_name in self._nodes}
            for agent_name in self._nodes:
                agent = self.agents[agent_name]
                agent.eval()
                with th.no_grad():
                    output = agent(batch, self.command)

                for metric_name, metric in self.metrics.items():
                    value = metric(input=output, target=target)
                    logs[agent_name][metric_name] = value

            for logger in self.loggers.values():
                logger.log({self.name: logs})
