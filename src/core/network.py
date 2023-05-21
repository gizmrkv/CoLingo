import networkx as nx


def generate_custom_graph(
    nodes: list | None = None, edges: list | None = None
) -> nx.DiGraph:
    g = nx.DiGraph()
    if nodes is not None:
        g.add_nodes_from(nodes)
    if edges is not None:
        g.add_edges_from(edges)
    return g


def generate_directed_complete_graph(
    nodes: list, allow_self_edges: bool = False
) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j or allow_self_edges:
                g.add_edge(nodes[i], nodes[j])
    return g
