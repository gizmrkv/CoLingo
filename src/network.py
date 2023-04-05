from abc import ABC


class Network(ABC):
    def __init__(self, edges: list[dict[str, str]]):
        self.edges = edges


class CustomNetwork(Network):
    def __init__(self, edges: list[dict[str, str]]):
        super().__init__(edges)


def build_network(network_type: str, network_args: dict) -> Network:
    networks_dict = {"custom": CustomNetwork}
    return networks_dict[network_type](**network_args)
