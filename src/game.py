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
        play_rate: float,
    ):
        self.dataloader = dataloader
        self.agents = agents
        self.network = network
        self.play_rate = play_rate

        self.play_pool = 0.0

    @abstractmethod
    def play(self):
        raise NotImplementedError


class LewisGame(Game):
    def __init__(
        self,
        agents: dict[str, Agent],
        network: Network,
        dataloader: DataLoader,
        play_rate: float,
    ):
        super().__init__(agents, network, dataloader, play_rate)

    def play(self):
        for count, batch in enumerate(self.dataloader):
            edge = random.choice(self.network.edges)
            sender = self.agents[edge["source"]]
            receiver = self.agents[edge["target"]]

            for agent in [sender, receiver]:
                agent.train()

            message = sender(batch, "object")
            answer = receiver(message, "message")
            reward = -th.nn.functional.cross_entropy(
                answer, batch, reduction="none"
            ).mean()

            for agent in [sender, receiver]:
                agent.optimizer.zero_grad()
                agent.loss(reward).backward(retain_graph=True)
                agent.optimizer.step()


def build_game(
    game_type: str,
    agents: dict[str, Agent],
    network: Network,
    dataloader: DataLoader,
    play_rate: float,
    game_args: dict,
) -> Game:
    games_dict = {"lewis": LewisGame}
    return games_dict[game_type](agents, network, dataloader, play_rate, **game_args)
