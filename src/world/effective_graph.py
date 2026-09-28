import math

import networkx as nx

from src.world.state import WorldState


def effective_subgraph(graph: nx.Graph, state: WorldState) -> nx.Graph:
    sub = graph.copy()
    for node_id in state.blocked_nodes:
        if sub.has_node(node_id):
            sub.remove_node(node_id)
    return sub


def congestion_penalty(u: str, v: str, graph: nx.Graph, state: WorldState) -> float:
    return 0.0


def danger_penalty(u: str, v: str, graph: nx.Graph, state: WorldState) -> float:
    return 0.0


def edge_cost(u: str, v: str, graph: nx.Graph, state: WorldState) -> float:
    base = graph.edges[u, v]["distance"]
    return base + congestion_penalty(u, v, graph, state) + danger_penalty(u, v, graph, state)


def heuristic(node_id: str, goal_id: str, graph: nx.Graph) -> float:
    x1, y1 = graph.nodes[node_id]["x"], graph.nodes[node_id]["y"]
    x2, y2 = graph.nodes[goal_id]["x"], graph.nodes[goal_id]["y"]
    return math.hypot(x2 - x1, y2 - y1)


def assert_nonnegative_penalties(graph: nx.Graph, state: WorldState) -> None:
    for u, v in graph.edges:
        assert congestion_penalty(u, v, graph, state) >= 0
        assert danger_penalty(u, v, graph, state) >= 0


def assert_base_distance_is_lower_bound(graph: nx.Graph) -> None:
    for u, v, data in graph.edges(data=True):
        assert data["distance"] >= heuristic(u, v, graph) - 1e-2


if __name__ == "__main__":
    from src.world.graph_loader import load_airport_graph

    g = load_airport_graph()
    state = WorldState().block_node("COR_6", reason="test")

    assert_base_distance_is_lower_bound(g)
    assert_nonnegative_penalties(g, state)
    print("graph_ok")

    print("edge_cost COR_1-COR_2:", edge_cost("COR_1", "COR_2", g, state))
    print("heuristic COR_1 to COR_10:", heuristic("COR_1", "COR_10", g))

    sub = effective_subgraph(g, state)
    print("nodes with COR_6 blocked:", sub.number_of_nodes())
    print("still connected:", nx.is_connected(sub))