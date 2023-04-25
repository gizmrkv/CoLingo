import os
import random
from abc import ABC, abstractmethod

import torch as th
from torch.utils.data import DataLoader

from .agent import Agent
from .network import Network


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

            message, aux_s = sender(batch, "object")
            answer, aux_r = receiver(message, "message")
            # FIX
            n_attributes = 2
            n_data = batch.shape[0]

            answer = answer.view(n_data * n_attributes, -1)
            labels = (
                batch.view(n_data, n_attributes, -1)
                .argmax(dim=-1)
                .view(n_data * n_attributes)
            )

            reward = (
                -th.nn.functional.cross_entropy(
                    answer,
                    labels,
                    reduction="none",
                )
                .view(-1, n_attributes)
                .mean()
            )

            sender.optimizer.zero_grad()
            sender_loss = sender.loss(reward, message, aux_s)
            sender_loss.backward(retain_graph=True)
            sender.optimizer.step()

            receiver.optimizer.zero_grad()
            receiver_loss = receiver.loss(reward, answer, aux_r)
            receiver_loss.backward(retain_graph=True)
            receiver.optimizer.step()
