from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class Rule:
    name: str
    condition: Callable
    action: str
    explanation: str


def has_incident(state, incident_type):
    return any(i.type == incident_type for i in state.active_incidents)


RULES = [
    Rule(
        "R1_fire_evacuation",
        lambda g, s: has_incident(s, "fire"),
        "activate_evacuation",
        "Fire detected, so evacuation is activated.",
    ),
    Rule(
        "R2_threat_dispatch",
        lambda g, s: s.threat_location is not None or has_incident(s, "threat"),
        "dispatch_security",
        "Threat detected, so security is dispatched.",
    ),
    Rule(
        "R3_corridor_reroute",
        lambda g, s: any(
            n in g.nodes and g.nodes[n]["type"] == "corridor" for n in s.blocked_nodes
        ),
        "recalculate_routes",
        "A corridor is blocked, so routes are recalculated.",
    ),
    Rule(
        "R4_unauthorized_alert",
        lambda g, s: has_incident(s, "unauthorized_access"),
        "trigger_security_alert",
        "Unauthorized person in a restricted zone, so a security alert is triggered.",
    ),
    Rule(
        "R5_high_severity_priority",
        lambda g, s: any(i.severity == "high" for i in s.active_incidents),
        "prioritize_emergency_response",
        "High severity incident, so emergency response is prioritized.",
    ),
    Rule(
        "R6_congestion_penalty",
        lambda g, s: any(
            n in g.nodes and count > 0.8 * g.nodes[n]["capacity"]
            for n, count in s.occupancy.items()
        ),
        "apply_congestion_penalty",
        "Passenger density is above 80% of capacity, so a congestion penalty is applied.",
    ),
]


def run_expert_system(graph, state) -> List[Dict]:
    fired = []
    for rule in RULES:
        if rule.condition(graph, state):
            fired.append(
                {"rule": rule.name, "action": rule.action, "explanation": rule.explanation}
            )
    return fired


if __name__ == "__main__":
    from src.world.graph_loader import load_airport_graph
    from src.world.scenario_loader import build_initial_state, load_scenarios

    g = load_airport_graph()
    for name in load_scenarios():
        s = build_initial_state(g, name)
        fired = run_expert_system(g, s)
        print(name)
        if not fired:
            print("  no rules fired")
        for f in fired:
            print(f"  {f['rule']} -> {f['action']}")