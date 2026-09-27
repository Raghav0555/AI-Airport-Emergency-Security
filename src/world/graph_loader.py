import csv
import os

import networkx as nx

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "airport"
)
NODES_CSV = os.path.join(DATA_DIR, "airport_nodes.csv")
EDGES_CSV = os.path.join(DATA_DIR, "airport_edges.csv")


def _str_to_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def load_airport_graph(nodes_path=NODES_CSV, edges_path=EDGES_CSV):
    graph = nx.Graph()

    with open(nodes_path, newline="") as f:
        for row in csv.DictReader(f):
            graph.add_node(
                row["node_id"],
                type=row["type"],
                x=float(row["x"]),
                y=float(row["y"]),
                capacity=int(row["capacity"]),
                zone_restricted=_str_to_bool(row["zone_restricted"]),
            )

    with open(edges_path, newline="") as f:
        for row in csv.DictReader(f):
            graph.add_edge(
                row["from_id"],
                row["to_id"],
                distance=float(row["distance"]),
                base_travel_time=float(row["base_travel_time"]),
                blocked=_str_to_bool(row["blocked"]),
            )

    return graph


def node_positions(graph):
    return {n: (d["x"], d["y"]) for n, d in graph.nodes(data=True)}


def nodes_by_type(graph, node_type):
    return [n for n, d in graph.nodes(data=True) if d["type"] == node_type]


if __name__ == "__main__":
    g = load_airport_graph()
    print(f"Loaded graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print("Connected:", nx.is_connected(g))
    for t in sorted({d["type"] for _, d in g.nodes(data=True)}):
        print(f"  {t}: {len(nodes_by_type(g, t))}")