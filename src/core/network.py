import networkx as nx


def create_custom_graph(
    nodes: list | None = None, edges: list | None = None
) -> nx.Graph:
    G = nx.Graph()
    if nodes is not None:
        G.add_nodes_from(nodes)
    if edges is not None:
        G.add_edges_from(edges)
    return G
