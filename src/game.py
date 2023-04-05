import random
import torch as th
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from .agent import Agent
from .network import Network


class Game(ABC):
    def __init__(
        self,
        agents: dict[str, Agent],
        network: Network,
        dataloader: DataLoader,
    ):
        self.dataloader = dataloader
        self.agents = agents
        self.network = network

    @abstractmethod
    def play(self) -> float:
        raise NotImplementedError


class LewisGame(Game):
    def __init__(
        self, agents: dict[str, Agent], network: Network, dataloader: DataLoader
    ):
        super().__init__(agents, network, dataloader)

    def play(self) -> float:
        reward_avg = 0
        for count, batch in enumerate(self.dataloader):
            edge = self.network.edges[random.randint(0, len(self.network.edges) - 1)]
            sender = self.agents[edge["source"]]
            receiver = self.agents[edge["target"]]

            message = sender(batch, "object", True)
            answer = receiver(message, "message", True)
            reward = -th.nn.functional.cross_entropy(
                answer, batch, reduction="none"
            ).mean()
            reward_avg += (reward.detach().item() - reward_avg) / (count + 1)

            for agent in [sender, receiver]:
                agent.optimizer.zero_grad()
                agent.loss(reward).backward(retain_graph=True)
                agent.optimizer.step()

        return reward_avg


def build_game(
    game_type: str,
    agents: dict[str, Agent],
    network: Network,
    dataloader: DataLoader,
    game_args: dict,
) -> Game:
    games_dict = {"lewis": LewisGame}
    return games_dict[game_type](agents, network, dataloader, **game_args)
