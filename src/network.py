class CustomNetwork:
    def __init__(self, edges: list, shuffle=True):
        self.edges = edges


def build_network(network_type: str, network_args: dict):
    networks_dict = {"custom": CustomNetwork}
    return networks_dict[network_type](**network_args)
