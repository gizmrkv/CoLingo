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
from ..core.network import generate_directed_complete_graph


class SignalTrainer(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        dataloader: DataLoader,
        sender_loss: th.nn.Module,
        receiver_loss: th.nn.Module,
        max_batches: int = 1,
        network: DiGraph | None = None,
        sender_command: Command = Command.SEND,
        receiver_command: Command = Command.RECEIVE,
    ):
        super().__init__()

        self.agents = agents
        self.dataloader = dataloader
        self.sender_loss = sender_loss
        self.receiver_loss = receiver_loss
        self.max_batches = max_batches
        self.network = network
        self.sender_command = sender_command
        self.receiver_command = receiver_command

        if self.network is None:
            self.network = generate_directed_complete_graph(list(self.agents.keys()))

        self._edges = list(self.network.edges)

    def on_update(self):
        for batch, target in islice(self.dataloader, self.max_batches):
            edge = random.choice(self._edges)
            sender = self.agents[edge[0]]
            receiver = self.agents[edge[1]]

            for agent in [sender, receiver]:
                agent.train()

            message, aux_s = sender(batch, self.sender_command)
            answer, aux_r = receiver(message, self.receiver_command)

            receiver_loss = self.receiver_loss(answer, target)
            sender_loss = self.sender_loss(receiver_loss, aux_s)

            loss = (sender_loss + receiver_loss).mean()

            sender.optimizer.zero_grad()
            receiver.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            sender.step()
            receiver.step()


class SignalEvaluator(Callback):
    def __init__(
        self,
        agents: dict[str, Agent],
        dataloader: DataLoader,
        metrics: dict[str, Callable],
        loggers: Iterable[Logger],
        name: str,
        interval: int = 1,
        network: DiGraph | None = None,
        sender_command: Command = Command.SEND,
        receiver_command: Command = Command.RECEIVE,
    ):
        super().__init__()
        self.agents = agents
        self.dataloader = dataloader
        self.metircs = metrics
        self.loggers = loggers
        self.network = network
        self.name = name
        self.interval = interval
        self.sender_command = sender_command
        self.receiver_command = receiver_command

        if self.network is None:
            self.network = generate_directed_complete_graph(list(self.agents.keys()))

        self._edges = list(self.network.edges)
        self._count = 0

    def on_update(self):
        if self._count % self.interval != 0:
            return

        self._count += 1

        for batch, target in self.dataloader:
            logs = {f"{edge[0]} -> {edge[1]}": {} for edge in self._edges}
            for edge in self._edges:
                sender = self.agents[edge[0]]
                receiver = self.agents[edge[1]]

                for agent in [sender, receiver]:
                    agent.eval()

                with th.no_grad():
                    message, aux_s = sender(batch, self.sender_command)
                    output, aux_r = receiver(message, self.receiver_command)

                for metric_name, metric in self.metircs.items():
                    value = metric(
                        input=output,
                        message=message,
                        target=target,
                        aux_s=aux_s,
                        aux_r=aux_r,
                    )
                    logs[f"{edge[0]} -> {edge[1]}"][metric_name] = value

            for logger in self.loggers:
                logger.log({self.name: logs})
