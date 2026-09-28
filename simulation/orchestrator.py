from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional

import networkx as nx

from src.astar.astar import astar
from src.csp.constraints import REQUIRED_AGENT_TYPE, validate_assignment, validate_route
from src.expert_system.rules import run_expert_system
from src.genetic_algorithm.ga import GAConfig, agent_location, run_ga
from src.minimax.minimax import simulate
from src.world.state import WorldState


def empty_csp_report():
    return {
        "routes_checked": 0,
        "assignments_checked": 0,
        "violations": [],
        "access_denied": [],
    }


@dataclass
class SimulationResult:
    scenario_name: str
    state: WorldState
    route_table: List[Dict] = field(default_factory=list)
    csp_report: Dict = field(default_factory=empty_csp_report)
    ga_assignment: Optional[Dict] = None
    minimax_outcome: Optional[Dict] = None
    expert_system_trace: List[Dict] = field(default_factory=list)
    face_recognition_result: Optional[Dict] = None
    decision_log: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.decision_log.append(message)


ActionHandler = Callable[[nx.Graph, WorldState, SimulationResult], None]

ACTION_HANDLERS: Dict[str, ActionHandler] = {}


def register_action(action_name: str):
    def decorator(fn: ActionHandler):
        ACTION_HANDLERS[action_name] = fn
        return fn

    return decorator


def agents_of(state, kind):
    return dict(getattr(state, f"{kind}_locations"))


def nodes_of_type(graph, node_type):
    return [n for n, d in graph.nodes(data=True) if d["type"] == node_type]


def best_exit_route(graph, state, start):
    options = [astar(graph, state, start, e) for e in nodes_of_type(graph, "exit")]
    options = [o for o in options if o.found]
    return min(options, key=lambda o: o.cost) if options else None


def staging_point(graph, state, node):
    if node not in state.blocked_nodes:
        return node
    open_neighbors = [n for n in graph.neighbors(node) if n not in state.blocked_nodes]
    if not open_neighbors:
        return node
    return min(open_neighbors, key=lambda n: graph.edges[node, n]["distance"])


def add_route(result, graph, state, agent, purpose, start, route, old=None, require_exit=False):
    found = route is not None and route.found
    row = {
        "agent": agent,
        "purpose": purpose,
        "start": start,
        "destination": route.path[-1] if found else "-",
        "path": route.path if found else [],
        "distance": round(route.cost, 1) if found else None,
        "old_distance": round(old.cost, 1) if old is not None and old.found else None,
        "status": "ok",
    }
    if not found:
        row["status"] = "unreachable"
    elif old is not None and old.found and old.path != route.path:
        row["status"] = "rerouted"

    if found:
        result.csp_report["routes_checked"] += 1
        result.csp_report["violations"].extend(
            validate_route(graph, state, agent, route.path, require_exit=require_exit)
        )
    result.route_table.append(row)
    return row


def assign_responders(graph, state, result):
    incidents = [
        replace(i, location_node=staging_point(graph, state, i.location_node))
        for i in state.active_incidents
        if i.type in REQUIRED_AGENT_TYPE
    ]
    if not incidents:
        return

    ga = run_ga(graph, state, incidents=incidents, config=GAConfig())

    rows = []
    for agent_id, incident in ga.assignment:
        start = agent_location(state, agent_id)
        route = astar(graph, state, start, incident.location_node)
        rows.append(
            {
                "incident": incident.incident_id,
                "type": incident.type,
                "severity": incident.severity,
                "agent": agent_id,
                "destination": incident.location_node,
            }
        )
        add_route(result, graph, state, agent_id, f"respond to {incident.incident_id}", start, route)
        result.log(
            f"GA: {agent_id} assigned to {incident.incident_id} ({incident.type}, {incident.severity})"
        )

    assignment_violations = validate_assignment(state, ga.assignment)
    result.csp_report["assignments_checked"] += len(ga.assignment)
    result.csp_report["violations"].extend(assignment_violations)

    result.ga_assignment = {
        "rows": rows,
        "best_fitness": round(ga.best_fitness, 2),
        "best_history": [round(v, 2) for v in ga.best_history],
        "avg_history": [round(v, 2) for v in ga.avg_history],
        "generations": len(ga.best_history),
        "astar_calls": ga.astar_calls,
        "time_ms": round(ga.time_ms, 1),
    }
    result.log(
        f"GA: best fitness {result.ga_assignment['best_fitness']} after "
        f"{result.ga_assignment['generations']} generations"
    )


@register_action("recalculate_routes")
def handle_recalculate_routes(graph, state, result):
    blocked_corridors = [n for n in sorted(state.blocked_nodes) if graph.nodes[n]["type"] == "corridor"]
    old_state = state
    for node in blocked_corridors:
        old_state = old_state.unblock_node(node)

    trips, affected, extra = 0, 0, []
    for kind in ("security", "emergency"):
        for agent_id, start in agents_of(state, kind).items():
            for terminal in nodes_of_type(graph, "terminal"):
                old = astar(graph, old_state, start, terminal)
                new = astar(graph, state, start, terminal)
                trips += 1
                if (not new.found) or (old.found and old.path != new.path):
                    affected += 1
                    add_route(result, graph, state, agent_id, f"response route to {terminal}", start, new, old=old)
                    if new.found and old.found:
                        extra.append(new.cost - old.cost)

    average_extra = round(sum(extra) / len(extra), 1) if extra else 0.0
    result.log(
        f"A*: blocked {blocked_corridors}; {affected} of {trips} response routes rerouted, "
        f"average extra distance {average_extra}"
    )


@register_action("activate_evacuation")
def handle_evacuation(graph, state, result):
    routed, unreachable = 0, 0
    for agent_id, start in sorted(agents_of(state, "passenger").items()):
        best = best_exit_route(graph, state, start)
        add_route(result, graph, state, agent_id, "evacuation to nearest exit", start, best, require_exit=True)
        if best is None:
            unreachable += 1
        else:
            routed += 1
    result.log(f"A*: {routed} passengers routed to exits, {unreachable} unreachable")


@register_action("dispatch_security")
def handle_dispatch_security(graph, state, result):
    threat_node = state.threat_location
    if threat_node is None:
        threat_node = next(i.location_node for i in state.active_incidents if i.type == "threat")

    minimax_run = simulate(graph, state, "minimax", "minimax", threat_start=threat_node)
    chase_run = simulate(graph, state, "chase", "minimax", threat_start=threat_node)

    result.minimax_outcome = {
        "threat_start": threat_node,
        "outcome": minimax_run.outcome,
        "rounds": minimax_run.rounds,
        "peak_risk": round(minimax_run.peak_risk, 1),
        "final_risk": round(minimax_run.final_risk, 1),
        "zones_closed": minimax_run.zones_closed,
        "nodes_evaluated": minimax_run.nodes_evaluated,
        "log": minimax_run.log,
        "chase_baseline": {
            "outcome": chase_run.outcome,
            "rounds": chase_run.rounds,
            "peak_risk": round(chase_run.peak_risk, 1),
        },
    }
    result.log(
        f"Minimax: threat at {threat_node} -> {minimax_run.outcome} in {minimax_run.rounds} rounds "
        f"(chase baseline: {chase_run.outcome})"
    )


@register_action("trigger_security_alert")
def handle_security_alert(graph, state, result):
    passengers = sorted(agents_of(state, "passenger").items())
    if not passengers:
        return
    intruder_id, start = passengers[-1]

    for incident in state.active_incidents:
        if incident.type != "unauthorized_access":
            continue
        route = astar(graph, state, start, incident.location_node)
        reasons = validate_route(graph, state, intruder_id, route.path) if route.found else ["no route"]
        result.csp_report["access_denied"].append(
            {
                "agent": intruder_id,
                "target": incident.location_node,
                "path": route.path,
                "allowed": not reasons,
                "reasons": reasons,
            }
        )
        verdict = "denied" if reasons else "allowed"
        result.log(f"CSP: access attempt by {intruder_id} to {incident.location_node} {verdict}")
    result.log("Face recognition: not connected yet")


def run_simulation(graph: nx.Graph, state: WorldState, scenario_name: str) -> SimulationResult:
    result = SimulationResult(scenario_name=scenario_name, state=state)

    result.expert_system_trace = run_expert_system(graph, state)
    for fired in result.expert_system_trace:
        result.log(f"Expert System: {fired['rule']} -> {fired['explanation']}")

    if not state.active_incidents:
        result.log("No active incidents - normal operation.")
        for agent_id, start in sorted(agents_of(state, "passenger").items()):
            best = best_exit_route(graph, state, start)
            add_route(result, graph, state, agent_id, "normal route to nearest exit", start, best)
        return result

    assign_responders(graph, state, result)

    for fired in result.expert_system_trace:
        handler = ACTION_HANDLERS.get(fired["action"])
        if handler is not None:
            handler(graph, state, result)

    return result


if __name__ == "__main__":
    from src.world.graph_loader import load_airport_graph
    from src.world.scenario_loader import build_initial_state, load_scenarios

    g = load_airport_graph()
    for name in load_scenarios():
        s = build_initial_state(g, name)
        r = run_simulation(g, s, name)
        print(f"=== {name}")
        for line in r.decision_log:
            print("  log:", line)
        for row in r.route_table[:4]:
            print("  route:", row["agent"], row["purpose"], row["start"], "->", row["destination"], row["distance"], row["status"])
        print("  routes:", len(r.route_table), "| csp violations:", r.csp_report["violations"], "| access_denied:", [(a["agent"], a["allowed"], a["reasons"]) for a in r.csp_report["access_denied"]])