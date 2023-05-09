import random

import torch as th
from networkx import Graph
from torch.utils.data import DataLoader

from ..core.agent import Agent
from ..core.callback import Callback
from ..core.logger import Logger


class IdentityTrainer(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        network: Graph,
        dataloader: DataLoader,
        loss: th.nn.Module,
        name: str,
    ):
        super().__init__()

        self.agents = agents
        self.network = network
        self.dataloader = dataloader
        self.loss = loss
        self.name = name

        self._nodes = list(self.network.nodes)

    def on_update(self):
        for _, batch in enumerate(self.dataloader):
            agent = self.agents[random.choice(self._nodes)]
            agent.train()
            output = agent(batch, "identity")
            loss = self.loss(output, batch).mean()
            agent.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            agent.optimizer.step()


class IdentityEvaluator(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        network: Graph,
        dataset: th.Tensor,
        metrics: dict[str, callable],
        loggers: list[Logger],
        interval: int = 10,
        name: str = "eval",
    ):
        super().__init__()
        self.agents = agents
        self.network = network
        self.dataset = dataset
        self.metrics = metrics
        self.loggers = loggers
        self.interval = interval
        self.name = name

        self._nodes = list(self.network.nodes)
        self._count = 0

    def on_update(self):
        if self._count % self.interval != 0:
            return

        self._count += 1

        logs = {agent_name: {} for agent_name in self._nodes}
        for agent_name in self._nodes:
            agent = self.agents[agent_name]
            agent.eval()
            with th.no_grad():
                output = agent(self.dataset, "identity")

            for metric_name, metric in self.metrics.items():
                value = metric(output, self.dataset)
                logs[agent_name][metric_name] = value

        for logger in self.loggers:
            logger.log({self.name: logs})
