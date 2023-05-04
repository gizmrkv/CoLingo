import os
import random
from abc import ABC, abstractmethod

import torch as th
from torch.utils.data import DataLoader

from .agent import Agent
from .network import Network
from .util import find_length
from .baseline import MeanBaseline


class Task(ABC):
    @abstractmethod
    def run(self):
        raise NotImplementedError


class AgentSaver(Task):
    def __init__(
        self,
        agents: dict[str, Agent],
        interval: float,
        path: str,
    ):
        super().__init__()

        self.agents = agents
        self.interval = interval
        self.path = path

        self.count = 0

    def run(self):
        if self.count % self.interval == 0:
            for agent_name, agent in self.agents.items():
                save_dir = f"{self.path}/{agent_name}"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                th.save(
                    agent,
                    f"{save_dir}/{self.count}.pth",
                )

        self.count += 1


class CommunicationTraining(Task):
    def __init__(self, network: Network, dataloader: DataLoader, name: str):
        super().__init__()

        self.network = network
        self.dataloader = dataloader
        self.name = name
        self.length_baseline = MeanBaseline()
        self.loss_baseline = MeanBaseline()

    def run(self):
        for _, batch in enumerate(self.dataloader):
            edge = random.choice(self.network.edges)
            sender = self.network.agents[edge["source"]]
            receiver = self.network.agents[edge["target"]]

            for agent in [sender, receiver]:
                agent.train()

            message, aux_s = sender(batch, "object")
            answer, aux_r = receiver(message, "message")

            receiver_loss = receiver.tasks[self.name]["loss"](answer, batch)
            sender_loss = sender.tasks[self.name]["loss"](receiver_loss, aux_s)

            loss = (sender_loss + receiver_loss).mean()

            sender.optimizer.zero_grad()
            receiver.optimizer.zero_grad()
            loss.backward()
            sender.optimizer.step()
            receiver.optimizer.step()

            # sender.tasks[self.name]["optimizer"].zero_grad()
            # receiver.tasks[self.name]["optimizer"].zero_grad()
            # loss.backward()
            # sender.tasks[self.name]["optimizer"].step()
            # receiver.tasks[self.name]["optimizer"].step()

            # for agent, loss in zip([sender, receiver], [sender_loss, receiver_loss]):
            #     agent.tasks[self.name]["optimizer"].zero_grad()
            #     loss.mean().backward()
            #     agent.tasks[self.name]["optimizer"].step()
