import random
from itertools import islice

import torch as th
from networkx import DiGraph
from torch.utils.data import DataLoader

from ..core.agent import Agent
from ..core.callback import Callback
from ..core.logger import Logger


class SignalingTrainer(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        network: DiGraph,
        dataloader: DataLoader,
        sender_loss: th.nn.Module,
        receiver_loss: th.nn.Module,
        max_batches: int = 1,
        name: str | None = None,
    ):
        super().__init__()

        self.agents = agents
        self.network = network
        self.dataloader = dataloader
        self.sender_loss = sender_loss
        self.receiver_loss = receiver_loss
        self.max_batches = max_batches
        self.name = name

        self._edges = list(self.network.edges)

    def on_update(self):
        for batch in islice(self.dataloader, self.max_batches):
            edge = random.choice(self._edges)
            sender = self.agents[edge[0]]
            receiver = self.agents[edge[1]]

            for agent in [sender, receiver]:
                agent.train()

            message, aux_s = sender(batch, "sender")
            answer, aux_r = receiver(message, "receiver")

            receiver_loss = self.receiver_loss(answer, batch)
            sender_loss = self.sender_loss(receiver_loss, aux_s)

            loss = (sender_loss + receiver_loss).mean()

            sender.optimizer.zero_grad()
            receiver.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            sender.optimizer.step()
            receiver.optimizer.step()


class SignalingEvaluator(Callback):
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
        self.metircs = metrics
        self.loggers = loggers
        self.interval = interval
        self.name = name

        self._edges = list(self.network.edges)
        self._count = 0

    def on_update(self):
        if self._count % self.interval != 0:
            return

        self._count += 1

        logs = {f"{edge[0]} -> {edge[1]}": {} for edge in self._edges}
        for edge in self._edges:
            sender = self.agents[edge[0]]
            receiver = self.agents[edge[1]]

            for agent in [sender, receiver]:
                agent.eval()

            with th.no_grad():
                message, aux_s = sender(self.dataset, "sender")
                output, aux_r = receiver(message, "receiver")

            for metric_name, metric in self.metircs.items():
                value = metric(
                    input=output,
                    message=message,
                    target=self.dataset,
                    aux_s=aux_s,
                    aux_r=aux_r,
                )
                logs[f"{edge[0]} -> {edge[1]}"][metric_name] = value

        for logger in self.loggers.values():
            logger.log({self.name: logs})
