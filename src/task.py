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
            # FIX
            n_attributes = 2
            n_data = batch.shape[0]

            answer = answer.view(n_data * n_attributes, -1)
            labels = (
                batch.view(n_data, n_attributes, -1)
                .argmax(dim=-1)
                .view(n_data * n_attributes)
            )

            loss = (
                th.nn.functional.cross_entropy(
                    answer,
                    labels,
                    reduction="none",
                )
                .view(-1, n_attributes)
                .mean(dim=-1)
                # .mean()
            )

            message_lengths = find_length(message)

            weighted_entropy = aux_s.entropy.mean() * 0.5 + aux_r.entropy.mean() * 0.5
            log_prob = aux_s.log_prob + aux_r.log_prob
            length_loss = message_lengths.float() * 0.0

            policy_length_loss = (
                (length_loss - self.length_baseline(None)) * aux_s.log_prob
            ).mean()
            policy_loss = ((loss.detach() - self.loss_baseline(None)) * log_prob).mean()

            optimized_loss = (
                policy_loss + policy_length_loss - weighted_entropy + loss.mean()
            )

            if True:
                self.length_baseline.update(None, length_loss)
                self.loss_baseline.update(None, loss)

            sender.tasks[self.name]["optimizer"].zero_grad()
            receiver.tasks[self.name]["optimizer"].zero_grad()
            optimized_loss.backward()
            sender.tasks[self.name]["optimizer"].step()
            receiver.tasks[self.name]["optimizer"].step()
