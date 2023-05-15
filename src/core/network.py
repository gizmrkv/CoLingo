import networkx as nx


def create_custom_graph(
    nodes: list | None = None, edges: list | None = None
) -> nx.DiGraph:
    g = nx.DiGraph()
    if nodes is not None:
        g.add_nodes_from(nodes)
    if edges is not None:
        g.add_edges_from(edges)
    return g
