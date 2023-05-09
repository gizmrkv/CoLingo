import random

import torch as th
from networkx import Graph
from torch.utils.data import DataLoader

from ..core.agent import Agent
from ..core.callback import Callback
from ..core.logger import Logger


class SignalingTrainer(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        network: Graph,
        dataloader: DataLoader,
        sender_loss: th.nn.Module,
        receiver_loss: th.nn.Module,
        name: str,
    ):
        super().__init__()

        self.agents = agents
        self.network = network
        self.dataloader = dataloader
        self.sender_loss = sender_loss
        self.receiver_loss = receiver_loss
        self.name = name

        self._edges = list(self.network.edges)

    def on_update(self):
        for _, batch in enumerate(self.dataloader):
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
        sender: th.nn.Module,
        receiver: th.nn.Module,
        dataset: th.Tensor,
        evaluator: dict,
        loggers: list[Logger],
        interval: int = 10,
        name: str = "eval",
    ):
        super().__init__()
        self.sender = sender
        self.receiver = receiver
        self.dataset = dataset
        self.evaluator = evaluator
        self.loggers = loggers
        self.interval = interval
        self.name = name

        self._count = 0

    def on_update(self):
        if self._count % self.interval != 0:
            return

        self._count += 1

        for agent in [self.sender, self.receiver]:
            agent.eval()
        message, aux_s = self.sender(self.dataset, "sender")
        output, aux_r = self.receiver(message, "receiver")
        logs = {}
        for name, func in self.evaluator.items():
            value = func(self.dataset, message, output, aux_s, aux_r)
            logs[name] = value

        for logger in self.loggers:
            logger.log({self.name: logs})
