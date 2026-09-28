REQUIRED_AGENT_TYPE = {
    "threat": "security",
    "unauthorized_access": "security",
    "fire": "emergency",
    "medical": "emergency",
    "corridor_blocked": "staff",
}

RESTRICTED_ALLOWED = {"security", "emergency", "staff"}

AGENT_KINDS = ("passenger", "security", "emergency", "staff")


def agent_type_of(agent_id, state):
    for kind in AGENT_KINDS:
        if agent_id in getattr(state, f"{kind}_locations"):
            return kind
    return None


def validate_route(graph, state, agent_id, route, require_exit=False):
    violations = []
    agent_type = agent_type_of(agent_id, state)

    for a, b in zip(route, route[1:]):
        if not graph.has_edge(a, b):
            violations.append(f"Route rejected: no corridor between {a} and {b}")

    for node in route:
        if node in state.blocked_nodes:
            violations.append(f"Route rejected: blocked node {node}")
        if node in state.restricted_zones and agent_type not in RESTRICTED_ALLOWED:
            violations.append(f"Route rejected: restricted zone {node} for {agent_id}")
        if state.occupancy.get(node, 0) >= graph.nodes[node]["capacity"]:
            violations.append(f"Route rejected: {node} is at full capacity")

    if require_exit and (not route or graph.nodes[route[-1]]["type"] != "exit"):
        violations.append("Route rejected: evacuation route must end at an exit")

    return violations


def validate_assignment(state, assignment):
    violations = []
    seen = {}

    for agent_id, incident in assignment:
        if agent_id in seen:
            violations.append(
                f"Assignment invalid: {agent_id} already assigned to {seen[agent_id]}"
            )
        seen[agent_id] = incident.incident_id

        required = REQUIRED_AGENT_TYPE.get(incident.type)
        actual = agent_type_of(agent_id, state)
        if required and actual != required:
            violations.append(
                f"Assignment invalid: {incident.incident_id} ({incident.type}) "
                f"needs a {required} agent, got {actual} {agent_id}"
            )

    assigned_ids = {incident.incident_id for _, incident in assignment}
    for incident in state.active_incidents:
        if incident.type in ("threat", "unauthorized_access") and incident.incident_id not in assigned_ids:
            violations.append(
                f"Assignment invalid: security incident {incident.incident_id} has no security response"
            )

    return violations


def solve_assignment(state, incidents):
    agents_by_type = {
        kind: list(getattr(state, f"{kind}_locations")) for kind in AGENT_KINDS
    }

    def backtrack(index, used, chosen):
        if index == len(incidents):
            return list(chosen)
        incident = incidents[index]
        kind = REQUIRED_AGENT_TYPE.get(incident.type)
        for agent_id in agents_by_type.get(kind, []):
            if agent_id in used:
                continue
            used.add(agent_id)
            chosen.append((agent_id, incident))
            result = backtrack(index + 1, used, chosen)
            if result is not None:
                return result
            chosen.pop()
            used.remove(agent_id)
        return None

    return backtrack(0, set(), [])


if __name__ == "__main__":
    from src.world.graph_loader import load_airport_graph
    from src.world.scenario_loader import build_initial_state

    g = load_airport_graph()
    s2 = build_initial_state(g, "scenario_2_blocked_corridor")
    s3 = build_initial_state(g, "scenario_3_security_threat")
    s5 = build_initial_state(g, "scenario_5_unauthorized_access")

    inc2 = s3.active_incidents[0]
    inc4 = s5.active_incidents[0]

    print("PAX_2 restricted:", validate_route(g, s3, "PAX_2", ["G2_3", "TERM_2", "RESTRICTED_1"]))
    print("SEC_2 restricted:", validate_route(g, s3, "SEC_2", ["CP_2", "TERM_2", "RESTRICTED_1"]))
    print("PAX_1 blocked:", validate_route(g, s2, "PAX_1", ["COR_5", "COR_6", "COR_7"]))
    print("no exit:", validate_route(g, s3, "PAX_1", ["G1_1", "TERM_1"], require_exit=True))
    print("double booking:", validate_assignment(s3, [("SEC_1", inc2), ("SEC_1", inc4)]))
    print("wrong type:", validate_assignment(s3, [("EMS_TEAM_1", inc2)]))

    solution = solve_assignment(s3, [inc2, inc4])
    print("solver:", [(a, i.incident_id) for a, i in solution])
    print("solver check:", validate_assignment(s3, solution))