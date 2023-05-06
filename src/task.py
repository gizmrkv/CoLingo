import os
import random

import torch as th
from torch.utils.data import DataLoader

from .agent import Agent
from .callback import Callback
from .network import Network


class AgentSaver(Callback):
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

    def on_update(self):
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


class SignalingTrainer(Callback):
    def __init__(
        self,
        network: Network,
        dataloader: DataLoader,
        sender_loss: th.nn.Module,
        receiver_loss: th.nn.Module,
        name: str,
    ):
        super().__init__()

        self.network = network
        self.dataloader = dataloader
        self.sender_loss = sender_loss
        self.receiver_loss = receiver_loss
        self.name = name

    def on_update(self):
        for _, batch in enumerate(self.dataloader):
            edge = random.choice(self.network.edges)
            sender = self.network.agents[edge["source"]]
            receiver = self.network.agents[edge["target"]]

            for agent in [sender, receiver]:
                agent.train()

            message, aux_s = sender(batch, "sender")
            answer, aux_r = receiver(message, "receiver")

            receiver_loss = self.receiver_loss(answer, batch)
            sender_loss = self.sender_loss(receiver_loss, aux_s)

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
