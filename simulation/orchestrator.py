from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import networkx as nx

from src.world.state import WorldState


@dataclass
class SimulationResult:
    scenario_name: str
    state: WorldState
    routes: Dict[str, List[str]] = field(default_factory=dict)
    csp_report: Optional[dict] = None
    ga_assignment: Optional[dict] = None
    minimax_outcome: Optional[dict] = None
    expert_system_trace: List[str] = field(default_factory=list)
    face_recognition_result: Optional[dict] = None
    decision_log: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.decision_log.append(message)


TriggerHandler = Callable[[nx.Graph, WorldState, SimulationResult], SimulationResult]

TRIGGERS: Dict[str, TriggerHandler] = {}


def register_trigger(incident_type: str):
    def decorator(fn: TriggerHandler):
        TRIGGERS[incident_type] = fn
        return fn

    return decorator


@register_trigger("corridor_blocked")
def handle_corridor_blocked(graph, state, result):
    result.log(
        f"[stub] corridor_blocked trigger fired, blocked_nodes={sorted(state.blocked_nodes)}"
    )
    return result


@register_trigger("threat")
def handle_threat(graph, state, result):
    result.log(f"[stub] threat trigger fired at {state.threat_location}")
    return result


@register_trigger("fire")
def handle_fire(graph, state, result):
    result.log("[stub] fire trigger fired")
    return result


@register_trigger("unauthorized_access")
def handle_unauthorized_access(graph, state, result):
    result.log("[stub] unauthorized_access trigger fired")
    return result


def run_simulation(graph: nx.Graph, state: WorldState, scenario_name: str) -> SimulationResult:
    result = SimulationResult(scenario_name=scenario_name, state=state)

    if not state.active_incidents:
        result.log("No active incidents - normal operation, no triggers fired.")
        return result

    fired = set()
    for incident in state.active_incidents:
        handler = TRIGGERS.get(incident.type)
        if handler is None:
            result.log(f"No handler registered for incident type '{incident.type}'")
            continue
        if incident.type in fired:
            continue
        fired.add(incident.type)
        result = handler(graph, state, result)

    return result


if __name__ == "__main__":
    from src.world.graph_loader import load_airport_graph
    from src.world.scenario_loader import build_initial_state, load_scenarios

    g = load_airport_graph()
    for name in load_scenarios():
        s = build_initial_state(g, name)
        r = run_simulation(g, s, name)
        print(f"{name}: {r.decision_log}")