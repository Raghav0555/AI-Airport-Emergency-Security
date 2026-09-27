
import csv
import math
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

NODES = {}


def add_node(node_id, node_type, x, y, capacity, restricted=False):
    NODES[node_id] = {
        "node_id": node_id,
        "type": node_type,
        "x": x,
        "y": y,
        "capacity": capacity,
        "zone_restricted": restricted,
    }


# --- Backbone corridor ---
BACKBONE_X_STEP = 100
for i in range(1, 11):  # COR_1 .. COR_10
    add_node(f"COR_{i}", "corridor", x=i * BACKBONE_X_STEP, y=0, capacity=200)

# --- Checkpoints ---
add_node("CP_1", "checkpoint", x=200, y=60, capacity=100)
add_node("CP_2", "checkpoint", x=500, y=60, capacity=100)
add_node("CP_3", "checkpoint", x=800, y=60, capacity=100)

# --- Terminals ---
add_node("TERM_1", "terminal", x=200, y=140, capacity=500)
add_node("TERM_2", "terminal", x=500, y=140, capacity=500)
add_node("TERM_3", "terminal", x=800, y=140, capacity=500)

# --- Gates (5 per terminal) ---
for term_num, term_x in [(1, 200), (2, 500), (3, 800)]:
    for g in range(1, 6):
        offset = (g - 3) * 35
        add_node(f"G{term_num}_{g}", "gate", x=term_x + offset, y=220, capacity=150)

# --- Emergency stations ---
add_node("EMS_1", "emergency_station", x=300, y=-60, capacity=20)
add_node("EMS_2", "emergency_station", x=700, y=-60, capacity=20)

# --- Restricted zone + staff area near Terminal 2 ---
add_node("RESTRICTED_1", "restricted", x=560, y=140, capacity=30, restricted=True)
add_node("STAFF_1", "staff_area", x=440, y=140, capacity=50, restricted=True)

# --- Restricted zone + staff area near Terminal 3 ---
add_node("RESTRICTED_2", "restricted", x=860, y=140, capacity=30, restricted=True)
add_node("STAFF_2", "staff_area", x=920, y=0, capacity=50, restricted=True)

# --- Emergency exits ---
add_node("EXIT_1", "exit", x=100, y=-140, capacity=999)
add_node("EXIT_2", "exit", x=500, y=-140, capacity=999)
add_node("EXIT_3", "exit", x=900, y=-140, capacity=999)

EDGES = []


def add_edge(a, b):
    EDGES.append((a, b))


# Backbone chain
for i in range(1, 10):
    add_edge(f"COR_{i}", f"COR_{i+1}")

# Redundancy shortcuts
add_edge("COR_2", "COR_6")
add_edge("COR_4", "COR_8")

# Checkpoint <-> backbone <-> terminal
add_edge("COR_2", "CP_1")
add_edge("CP_1", "TERM_1")
add_edge("COR_5", "CP_2")
add_edge("CP_2", "TERM_2")
add_edge("COR_8", "CP_3")
add_edge("CP_3", "TERM_3")

# Gates <-> terminal
for term_num in (1, 2, 3):
    for g in range(1, 6):
        add_edge(f"TERM_{term_num}", f"G{term_num}_{g}")

# Emergency stations <-> backbone
add_edge("COR_3", "EMS_1")
add_edge("COR_7", "EMS_2")

# Restricted zones / staff areas
add_edge("TERM_2", "RESTRICTED_1")
add_edge("TERM_2", "STAFF_1")
add_edge("TERM_3", "RESTRICTED_2")
add_edge("COR_9", "STAFF_2")

# Exits <-> backbone
add_edge("COR_1", "EXIT_1")
add_edge("COR_5", "EXIT_2")
add_edge("COR_10", "EXIT_3")


def euclidean(a_id, b_id):
    ax, ay = NODES[a_id]["x"], NODES[a_id]["y"]
    bx, by = NODES[b_id]["x"], NODES[b_id]["y"]
    return round(math.hypot(bx - ax, by - ay), 2)


def write_nodes_csv(path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["node_id", "type", "x", "y", "capacity", "zone_restricted"]
        )
        writer.writeheader()
        for node in NODES.values():
            writer.writerow(node)


def write_edges_csv(path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["from_id", "to_id", "distance", "base_travel_time", "blocked"]
        )
        writer.writeheader()
        for a, b in EDGES:
            dist = euclidean(a, b)
            travel_time = round(dist / 1.4, 2)
            writer.writerow(
                {
                    "from_id": a,
                    "to_id": b,
                    "distance": dist,
                    "base_travel_time": travel_time,
                    "blocked": False,
                }
            )


def main():
    for a, b in EDGES:
        assert a in NODES, f"Edge references unknown node: {a}"
        assert b in NODES, f"Edge references unknown node: {b}"

    nodes_path = os.path.join(OUT_DIR, "airport_nodes.csv")
    edges_path = os.path.join(OUT_DIR, "airport_edges.csv")
    write_nodes_csv(nodes_path)
    write_edges_csv(edges_path)

    print(f"Wrote {len(NODES)} nodes -> {nodes_path}")
    print(f"Wrote {len(EDGES)} edges -> {edges_path}")


if __name__ == "__main__":
    main()