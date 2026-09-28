from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.astar.astar import astar

INTERCEPT_REWARD = 1000.0
BREACH_PENALTY = 1000.0
PROXIMITY_RANGE = 500.0
ZONE_RISK_WEIGHT = 100.0
CLOSE_COST = 20.0
OFFICER_DISTANCE_WEIGHT = 0.1
FAR = 2000.0


@dataclass(frozen=True)
class GameState:
    threat: str
    officers: Tuple[Tuple[str, str], ...]
    closed: frozenset


@dataclass
class SimResult:
    outcome: str
    rounds: int
    peak_risk: float
    final_risk: float
    zones_closed: int
    nodes_evaluated: int
    log: List[Dict]


class GameContext:
    def __init__(self, graph, state, protected):
        self.graph = graph
        self.state = state
        self.protected = frozenset(protected)
        self.nodes_evaluated = 0
        self._dist = {}
        self._step = {}

    def _path(self, a, b):
        key = (a, b)
        if key not in self._dist:
            result = astar(self.graph, self.state, a, b)
            self._dist[key] = result.cost if result.found else FAR
            self._step[key] = result.path[1] if result.found and len(result.path) > 1 else a
        return key

    def dist(self, a, b):
        if a == b:
            return 0.0
        return self._dist[self._path(a, b)]

    def step_toward(self, a, b):
        if a == b:
            return a
        return self._step[self._path(a, b)]


def default_protected(graph, state, threat_start):
    exits = {n for n, d in graph.nodes(data=True) if d["type"] == "exit"}
    return (set(state.restricted_zones) | exits) - {threat_start} - set(state.blocked_nodes)


def zone_risk(gs, ctx):
    return sum(
        ZONE_RISK_WEIGHT * max(0.0, 1.0 - ctx.dist(gs.threat, z) / PROXIMITY_RANGE)
        for z in ctx.protected - gs.closed
    )


def evaluate(gs, ctx):
    min_officer_distance = min(ctx.dist(node, gs.threat) for _, node in gs.officers)
    return (
        -zone_risk(gs, ctx)
        - OFFICER_DISTANCE_WEIGHT * min_officer_distance
        - CLOSE_COST * len(gs.closed)
    )


def terminal_value(gs, ctx, depth):
    if gs.threat in {node for _, node in gs.officers}:
        return INTERCEPT_REWARD + depth
    if gs.threat in ctx.protected and gs.threat not in gs.closed:
        return -BREACH_PENALTY - depth
    return None


def max_actions(gs, ctx):
    actions = [("advance", officer_id) for officer_id, _ in gs.officers]
    actions += [("close", zone) for zone in sorted(ctx.protected - gs.closed)]
    return actions


def apply_max(gs, action, ctx):
    kind, target = action
    if kind == "advance":
        officers = tuple(
            (oid, ctx.step_toward(node, gs.threat) if oid == target else node)
            for oid, node in gs.officers
        )
        return GameState(gs.threat, officers, gs.closed)
    return GameState(gs.threat, gs.officers, gs.closed | {target})


def min_moves(gs, ctx):
    moves = [
        n
        for n in ctx.graph.neighbors(gs.threat)
        if n not in ctx.state.blocked_nodes and n not in gs.closed
    ]
    return sorted(moves) + [gs.threat]


def minimax(gs, depth, alpha, beta, maximizing, ctx, pruning=True):
    ctx.nodes_evaluated += 1

    value = terminal_value(gs, ctx, depth)
    if value is not None:
        return value
    if depth == 0:
        return evaluate(gs, ctx)

    if maximizing:
        best = float("-inf")
        for action in max_actions(gs, ctx):
            child = apply_max(gs, action, ctx)
            best = max(best, minimax(child, depth - 1, alpha, beta, False, ctx, pruning))
            alpha = max(alpha, best)
            if pruning and beta <= alpha:
                break
        return best

    best = float("inf")
    for move in min_moves(gs, ctx):
        child = GameState(move, gs.officers, gs.closed)
        best = min(best, minimax(child, depth - 1, alpha, beta, True, ctx, pruning))
        beta = min(beta, best)
        if pruning and beta <= alpha:
            break
    return best


def best_max_action(gs, depth, ctx, pruning=True):
    best_value, best_action = float("-inf"), None
    alpha = float("-inf")
    for action in max_actions(gs, ctx):
        child = apply_max(gs, action, ctx)
        value = minimax(child, depth - 1, alpha, float("inf"), False, ctx, pruning)
        if value > best_value:
            best_value, best_action = value, action
        alpha = max(alpha, best_value)
    return best_action, best_value


def best_min_move(gs, depth, ctx, pruning=True):
    best_value, best_move = float("inf"), None
    beta = float("inf")
    for move in min_moves(gs, ctx):
        child = GameState(move, gs.officers, gs.closed)
        value = minimax(child, depth - 1, float("-inf"), beta, True, ctx, pruning)
        if value < best_value:
            best_value, best_move = value, move
        beta = min(beta, best_value)
    return best_move, best_value


def greedy_threat_move(gs, ctx):
    open_zones = ctx.protected - gs.closed
    moves = [m for m in min_moves(gs, ctx) if m != gs.threat] or [gs.threat]
    if not open_zones:
        return moves[0]
    return min(moves, key=lambda m: (min(ctx.dist(m, z) for z in open_zones), m))


def chase_action(gs, ctx):
    officer_id = min(gs.officers, key=lambda o: (ctx.dist(o[1], gs.threat), o[0]))[0]
    return ("advance", officer_id)


def outcome_of(gs, ctx):
    value = terminal_value(gs, ctx, 0)
    if value is None:
        return None
    return "intercepted" if value > 0 else "breach"


def simulate(
    graph,
    state,
    defender="minimax",
    attacker="minimax",
    threat_start=None,
    depth=4,
    attacker_depth=3,
    max_rounds=8,
    pruning=True,
):
    start = threat_start or state.threat_location
    protected = default_protected(graph, state, start)
    ctx = GameContext(graph, state, protected)
    gs = GameState(start, tuple(sorted(state.security_locations.items())), frozenset())

    log = []
    outcome = "contained"
    peak_risk = zone_risk(gs, ctx)
    rounds = 0

    for round_number in range(1, max_rounds + 1):
        rounds = round_number

        if defender == "minimax":
            action, _ = best_max_action(gs, depth, ctx, pruning)
        elif defender == "chase":
            action = chase_action(gs, ctx)
        else:
            action = None
        if action is not None:
            gs = apply_max(gs, action, ctx)

        result = outcome_of(gs, ctx)
        if result:
            outcome = result
            log.append({"round": round_number, "security": action, "threat": gs.threat})
            break

        if attacker == "minimax":
            move, _ = best_min_move(gs, attacker_depth, ctx, pruning)
        else:
            move = greedy_threat_move(gs, ctx)
        gs = GameState(move, gs.officers, gs.closed)

        peak_risk = max(peak_risk, zone_risk(gs, ctx))
        log.append(
            {
                "round": round_number,
                "security": action,
                "threat": gs.threat,
                "risk": round(zone_risk(gs, ctx), 1),
            }
        )

        result = outcome_of(gs, ctx)
        if result:
            outcome = result
            break

    return SimResult(
        outcome=outcome,
        rounds=rounds,
        peak_risk=peak_risk,
        final_risk=zone_risk(gs, ctx),
        zones_closed=len(gs.closed),
        nodes_evaluated=ctx.nodes_evaluated,
        log=log,
    )


if __name__ == "__main__":
    from time import perf_counter

    from src.world.graph_loader import load_airport_graph
    from src.world.scenario_loader import build_initial_state

    g = load_airport_graph()
    state = build_initial_state(g, "scenario_3_security_threat")

    print("Scenario 3 trace (minimax defender vs minimax attacker)")
    trace = simulate(g, state, "minimax", "minimax")
    for entry in trace.log:
        print(" ", entry)
    print("  outcome:", trace.outcome, "| rounds:", trace.rounds)

    ctx_a = GameContext(g, state, default_protected(g, state, state.threat_location))
    ctx_b = GameContext(g, state, default_protected(g, state, state.threat_location))
    gs0 = GameState(
        "TERM_2", tuple(sorted(state.security_locations.items())), frozenset()
    )
    with_pruning = best_max_action(gs0, 4, ctx_a, pruning=True)
    without_pruning = best_max_action(gs0, 4, ctx_b, pruning=False)
    print("\nAlpha-beta check (threat at TERM_2, depth 4)")
    print("  with pruning:", with_pruning, "| nodes:", ctx_a.nodes_evaluated)
    print("  without pruning:", without_pruning, "| nodes:", ctx_b.nodes_evaluated)
    print("  same decision:", with_pruning == without_pruning)

    exits = {n for n, d in g.nodes(data=True) if d["type"] == "exit"}
    starts = sorted(
        n for n in g.nodes
        if n not in state.restricted_zones and n not in exits and n not in state.blocked_nodes
    )
    print(f"\nSweep: threat starts at each of {len(starts)} nodes, attacker = minimax")
    print(f"{'defender':<10} {'intercepted':>11} {'breach':>7} {'contained':>10} {'avg_peak_risk':>14} {'avg_rounds':>11}")
    t0 = perf_counter()
    for defender in ("idle", "chase", "minimax"):
        results = [simulate(g, state, defender, "minimax", threat_start=s) for s in starts]
        counts = {k: sum(r.outcome == k for r in results) for k in ("intercepted", "breach", "contained")}
        avg_peak = sum(r.peak_risk for r in results) / len(results)
        avg_rounds = sum(r.rounds for r in results) / len(results)
        print(f"{defender:<10} {counts['intercepted']:>11} {counts['breach']:>7} {counts['contained']:>10} {avg_peak:>14.1f} {avg_rounds:>11.2f}")
    print("sweep time s:", round(perf_counter() - t0, 1))