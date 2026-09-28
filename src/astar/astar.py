import heapq
import math
from dataclasses import dataclass
from itertools import count
from time import perf_counter
from typing import List

import networkx as nx

from src.world.effective_graph import edge_cost, heuristic
from src.world.state import WorldState


@dataclass
class AStarResult:
    found: bool
    path: List[str]
    cost: float
    nodes_expanded: int
    time_ms: float


def astar(graph: nx.Graph, state: WorldState, start: str, goal: str) -> AStarResult:
    t0 = perf_counter()

    if start in state.blocked_nodes or goal in state.blocked_nodes:
        return AStarResult(False, [], math.inf, 0, (perf_counter() - t0) * 1000)

    tie_breaker = count()
    open_heap = [(heuristic(start, goal, graph), next(tie_breaker), start)]
    g_score = {start: 0.0}
    came_from = {}
    closed = set()
    nodes_expanded = 0

    while open_heap:
        _, _, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        closed.add(node)
        nodes_expanded += 1

        if node == goal:
            path = [node]
            while node in came_from:
                node = came_from[node]
                path.append(node)
            path.reverse()
            return AStarResult(
                True, path, g_score[goal], nodes_expanded, (perf_counter() - t0) * 1000
            )

        for neighbor in graph.neighbors(node):
            if neighbor in state.blocked_nodes or neighbor in closed:
                continue
            new_g = g_score[node] + edge_cost(node, neighbor, graph, state)
            if new_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = new_g
                came_from[neighbor] = node
                f_score = new_g + heuristic(neighbor, goal, graph)
                heapq.heappush(open_heap, (f_score, next(tie_breaker), neighbor))

    return AStarResult(False, [], math.inf, nodes_expanded, (perf_counter() - t0) * 1000)


if __name__ == "__main__":
    from src.world.effective_graph import effective_subgraph
    from src.world.graph_loader import load_airport_graph

    g = load_airport_graph()

    normal = WorldState()
    blocked = WorldState().block_node("COR_6", reason="test")

    for label, state in [("normal", normal), ("COR_6 blocked", blocked)]:
        r = astar(g, state, "EXIT_1", "EXIT_3")
        sub = effective_subgraph(g, state)
        dijkstra_cost = nx.dijkstra_path_length(sub, "EXIT_1", "EXIT_3", weight="distance")
        print(f"[{label}]")
        print("  path:", " -> ".join(r.path))
        print("  cost:", round(r.cost, 2), "| nodes expanded:", r.nodes_expanded)
        print("  matches dijkstra:", abs(r.cost - dijkstra_cost) < 1e-6)