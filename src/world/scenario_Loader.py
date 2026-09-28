import csv
import json
import os

import networkx as nx

from src.world.state import Incident, WorldState

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
AGENTS_CSV = os.path.join(DATA_DIR, "agents", "agents.csv")
SCENARIOS_JSON = os.path.join(DATA_DIR, "scenarios", "scenarios.json")


def load_scenarios(path=SCENARIOS_JSON):
    with open(path) as f:
        return json.load(f)


def load_agents(path=AGENTS_CSV):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_initial_state(graph: nx.Graph, scenario_name: str) -> WorldState:
    scenarios = load_scenarios()
    if scenario_name not in scenarios:
        raise KeyError(
            f"Unknown scenario '{scenario_name}'. Available: {list(scenarios.keys())}"
        )
    scenario = scenarios[scenario_name]

    passenger_locations = {}
    security_locations = {}
    emergency_locations = {}
    staff_locations = {}
    permissions = {}

    location_maps = {
        "passenger": passenger_locations,
        "security": security_locations,
        "emergency": emergency_locations,
        "staff": staff_locations,
    }

    for row in load_agents():
        target = location_maps.get(row["type"])
        if target is not None:
            target[row["agent_id"]] = row["home_node"]
        permissions[row["agent_id"]] = row["authorization_level"]

    restricted_zones = {
        n for n, d in graph.nodes(data=True) if d.get("zone_restricted")
    }
    active_gates = {n for n, d in graph.nodes(data=True) if d.get("type") == "gate"}

    incidents = [
        Incident(
            incident_id=i["incident_id"],
            type=i["type"],
            location_node=i["location_node"],
            severity=i["severity"],
        )
        for i in scenario.get("active_incidents", [])
    ]

    return WorldState(
        passenger_locations=passenger_locations,
        security_locations=security_locations,
        emergency_locations=emergency_locations,
        staff_locations=staff_locations,
        threat_location=scenario.get("threat_location"),
        blocked_nodes=set(scenario.get("blocked_nodes", [])),
        active_gates=active_gates,
        restricted_zones=restricted_zones,
        occupancy={},
        permissions=permissions,
        active_incidents=incidents,
        event_log=[f"Loaded {scenario_name}: {scenario.get('description', '')}"],
    )


if __name__ == "__main__":
    from src.world.graph_loader import load_airport_graph

    g = load_airport_graph()
    for name in load_scenarios():
        s = build_initial_state(g, name)
        print(
            f"{name}: blocked={sorted(s.blocked_nodes)} "
            f"threat={s.threat_location} incidents={len(s.active_incidents)}"
        )