import re
from dataclasses import dataclass, replace
from typing import List

from simulation.orchestrator import best_exit_route


@dataclass
class Reply:
    intent: str
    text: str
    sources: List[str]


def fmt_path(path):
    return " -> ".join(path)


def current_route(graph, state, result, user_id):
    start = state.passenger_locations.get(user_id)
    if start is None:
        return None
    for row in result.route_table:
        if row["agent"] == user_id and row["path"] and "exit" in row["purpose"]:
            return row
    best = best_exit_route(graph, state, start)
    if best is None:
        return None
    return {"path": best.path, "destination": best.path[-1], "distance": round(best.cost, 1)}


def answer_route(ctx):
    graph, state, result, user_id = ctx
    route = current_route(graph, state, result, user_id)
    if route is None:
        return Reply("route", "I could not find a safe route to an exit from your location. Please ask airport staff for help.", ["A*"])
    text = (
        f"The nearest available exit is {route['destination']}. "
        f"Your route: {fmt_path(route['path'])} ({route['distance']} units)."
    )
    if state.blocked_nodes:
        text += f" Blocked areas ({', '.join(sorted(state.blocked_nodes))}) are avoided."
    sources = ["A* route planner"]
    if result.csp_report["routes_checked"]:
        sources.append("CSP constraint checker")
    return Reply("route", text, sources)


def answer_route_changed(ctx):
    graph, state, result, user_id = ctx
    start = state.passenger_locations.get(user_id)
    if start is None:
        return Reply("route_changed", "I do not know your location.", [])
    now = best_exit_route(graph, state, start)
    before = best_exit_route(graph, replace(state, blocked_nodes=set()), start)
    sources = ["A* route planner"]

    if not state.blocked_nodes:
        return Reply("route_changed", "Nothing is blocked right now, so your route has not changed.", sources)
    if now is None:
        return Reply("route_changed", f"Blocked areas ({', '.join(sorted(state.blocked_nodes))}) cut off every exit from your location. Please ask airport staff for help.", sources)
    if before.path == now.path:
        return Reply(
            "route_changed",
            f"Your route has not changed. The blocked areas ({', '.join(sorted(state.blocked_nodes))}) are not on your path: "
            f"{fmt_path(now.path)} ({round(now.cost, 1)} units).",
            sources,
        )
    hit = [n for n in before.path if n in state.blocked_nodes]
    return Reply(
        "route_changed",
        f"Your original route ({fmt_path(before.path)}, {round(before.cost, 1)} units) used blocked area(s) {', '.join(hit)}, "
        f"so the route planner recalculated it. New route: {fmt_path(now.path)} ({round(now.cost, 1)} units).",
        sources,
    )


def answer_restricted(ctx):
    graph, state, result, user_id = ctx
    zones = ", ".join(sorted(state.restricted_zones))
    text = (
        f"Restricted areas ({zones}) are only for authorised airport personnel: security, emergency responders and staff. "
        "Passenger routes through them are rejected by the constraint checker."
    )
    denied = [a for a in result.csp_report["access_denied"] if not a["allowed"]]
    if denied:
        text += f" Latest check: {denied[0]['reasons'][0]}."
    return Reply("restricted", text, ["CSP constraint checker"])


def answer_responders(ctx):
    graph, state, result, user_id = ctx
    ga = result.ga_assignment
    if not ga:
        return Reply("responders", "No responders are needed because there are no active incidents.", ["Genetic algorithm"])
    lines = [f"{r['agent']} -> {r['incident']} ({r['type']}, {r['severity']})" for r in ga["rows"]]
    text = "Responders: " + "; ".join(lines) + ". The assignment was optimised by the genetic algorithm and validated by the constraint checker."
    return Reply("responders", text, ["Genetic algorithm", "CSP constraint checker"])


def answer_threat(ctx):
    graph, state, result, user_id = ctx
    mm = result.minimax_outcome
    if not mm:
        return Reply("threat", "There is no active security threat.", ["Minimax"])
    base = mm["chase_baseline"]
    text = (
        f"A simulated threat at {mm['threat_start']} was {mm['outcome']} in {mm['rounds']} round(s) by the minimax defence. "
        f"A simple chase defence would have ended: {base['outcome']} in {base['rounds']} round(s)."
    )
    return Reply("threat", text, ["Minimax"])


def answer_blocked(ctx):
    graph, state, result, user_id = ctx
    if not state.blocked_nodes:
        return Reply("blocked", "No corridors or areas are blocked.", [])
    return Reply("blocked", f"Blocked areas: {', '.join(sorted(state.blocked_nodes))}. Routes avoid them automatically.", ["A* route planner"])


def answer_status(ctx):
    graph, state, result, user_id = ctx
    if not state.active_incidents:
        return Reply("status", "Everything is normal. There are no active incidents.", ["Expert system"])
    incidents = "; ".join(f"{i.type} at {i.location_node} (severity {i.severity})" for i in state.active_incidents)
    text = f"Active incidents: {incidents}."
    if result.expert_system_trace:
        first = result.expert_system_trace[0]
        text += f" Rule triggered: {first['rule']} -> {first['explanation']}"
    return Reply("status", text, ["Expert system"])


def answer_help(ctx):
    text = (
        "You can ask me: 'Where should I go?', 'Where is the emergency exit?', 'Why was my route changed?', "
        "'Why can't I enter this area?', 'What is happening?', 'Who is responding?', 'Is there a threat?' or 'What is blocked?'"
    )
    return Reply("help", text, [])


INTENTS = [
    ("route_changed", r"(why|how).*(rout|path).*(chang|differ|new)|reroute|re-route|why.*chang", answer_route_changed),
    ("restricted", r"restrict|(why|can|cant|can't|cannot|allowed|permit).*(enter|access|go in|go into)|not allowed|authori[sz]", answer_restricted),
    ("route", r"exit|where.*(go|run|head)|route|way out|evacuat|safe", answer_route),
    ("responders", r"respond|dispatch|assign|who.*(coming|helping)|security|officer|team", answer_responders),
    ("threat", r"threat|attacker|intruder|minimax|danger|suspicious", answer_threat),
    ("blocked", r"block|closed|corridor|obstruct", answer_blocked),
    ("status", r"happen|going on|incident|status|situation|emergency|fire|alert|wrong", answer_status),
]


def answer(question, graph, state, result, user_id):
    text = question.lower().strip()
    ctx = (graph, state, result, user_id)
    for intent, pattern, handler in INTENTS:
        if re.search(pattern, text):
            return handler(ctx)
    return answer_help(ctx)


if __name__ == "__main__":
    from simulation.orchestrator import run_simulation
    from src.world.graph_loader import load_airport_graph
    from src.world.scenario_loader import build_initial_state

    g = load_airport_graph()
    questions = [
        "Where should I go?",
        "Why was my route changed?",
        "Where is the emergency exit?",
        "Why can't I enter this area?",
        "What is happening?",
        "Who is responding?",
        "Is there a threat?",
        "What is blocked?",
        "hello",
    ]
    for scenario, user in [
        ("scenario_4_evacuation", "PAX_2"),
        ("scenario_3_security_threat", "PAX_2"),
        ("scenario_5_unauthorized_access", "PAX_4"),
    ]:
        s = build_initial_state(g, scenario)
        r = run_simulation(g, s, scenario)
        print(f"===== {scenario} (I am {user})")
        for q in questions:
            reply = answer(q, g, s, r, user)
            print(f"Q: {q}\n   [{reply.intent}] {reply.text}\n   sources: {reply.sources}")