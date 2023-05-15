import random
from itertools import islice

import torch as th
from networkx import DiGraph
from torch.utils.data import DataLoader

from ..core.agent import Agent
from ..core.callback import Callback
from ..core.logger import Logger


class IdentityTrainer(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        network: DiGraph,
        dataloader: DataLoader,
        loss: th.nn.Module,
        max_batches: int = 1,
        name: str | None = None,
    ):
        super().__init__()

        self.agents = agents
        self.network = network
        self.dataloader = dataloader
        self.loss = loss
        self.max_batches = max_batches
        self.name = name

        self._nodes = list(self.network.nodes)

    def on_update(self):
        for batch in islice(self.dataloader, self.max_batches):
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
        network: DiGraph,
        dataset: th.Tensor,
        metrics: dict[str, callable],
        loggers: dict[str, Logger],
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
                value = metric(input=output, target=self.dataset)
                logs[agent_name][metric_name] = value

        for logger in self.loggers.values():
            logger.log({self.name: logs})
