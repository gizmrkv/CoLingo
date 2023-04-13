from abc import ABC
from .agent import Agent


class Network(ABC):
    def __init__(self, agents: dict[str, Agent], edges: list[dict[str, str]]):
        self.agents = agents
        self.edges = edges


class CustomNetwork(Network):
    def __init__(self, agents: dict[str, Agent], edges: list[dict[str, str]]):
        super().__init__(agents, edges)


def build_network(network_type: str, network_args: dict) -> Network:
    networks_dict = {"custom": CustomNetwork}
    return networks_dict[network_type](**network_args)
