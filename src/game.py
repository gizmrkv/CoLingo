import random
import torch as th


class LewisGame:
    def __init__(self, dataloader, agents, network):
        self.dataloader = dataloader
        self.agents = agents
        self.network = network

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


def build_game(game_type: str, dataloader, agents, network, game_args: dict):
    games_dict = {"lewis": LewisGame}
    return games_dict[game_type](dataloader, agents, network, **game_args)
