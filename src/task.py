from abc import ABC, abstractmethod

import random
import torch as th
from torch.utils.data import DataLoader
from .network import Network


class Task(ABC):
    @abstractmethod
    def run(self):
        raise NotImplementedError


class LewisGame(Task):
    def __init__(
        self,
        network: Network,
        dataloader: DataLoader,
    ):
        super().__init__()

        self.network = network
        self.dataloader = dataloader

    def run(self):
        for _, batch in enumerate(self.dataloader):
            edge = random.choice(self.network.edges)
            sender = self.network.agents[edge["source"]]
            receiver = self.network.agents[edge["target"]]

            for agent in [sender, receiver]:
                agent.train()

            message ,aux_s= sender(batch, "object")
            answer,aux_r = receiver(message, "message")
            reward = -th.nn.functional.cross_entropy(
                answer, batch, reduction="none"
            ).mean()

            sender.optimizer.zero_grad()
            sender.loss(reward, message, aux_s).backward(retain_graph=True)
            sender.optimizer.step()

            receiver.optimizer.zero_grad()
            receiver.loss(reward, answer, aux_r).backward(retain_graph=True)
            receiver.optimizer.step()
