import csv
import json
import os
import random
from collections import deque
from dataclasses import replace
from heapq import heappop, heappush
from itertools import count
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.orchestrator import run_simulation
from src.astar.astar import astar
from src.csp.constraints import validate_assignment, validate_route
from src.expert_system.rules import run_expert_system
from src.genetic_algorithm.ga import (
    SEVERITY_WEIGHT,
    GAConfig,
    brute_force_best,
    build_domains,
    fitness,
    greedy_assignment,
    make_cost_fn,
    run_ga,
)
from src.minimax.minimax import GameContext, GameState, best_max_action, default_protected, simulate
from src.world.effective_graph import edge_cost
from src.world.graph_loader import load_airport_graph
from src.world.scenario_loader import build_initial_state, load_scenarios
from src.world.state import Incident

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "results")

CSP_TRIALS = 200
CSP_REQUESTS_PER_TRIAL = 5
GA_TRIALS = 100
ALL_ACTIONS = [
    "activate_evacuation",
    "dispatch_security",
    "recalculate_routes",
    "trigger_security_alert",
    "prioritize_emergency_response",
    "apply_congestion_penalty",
]


def nodes_of_type(graph, *types):
    return [n for n, d in graph.nodes(data=True) if d["type"] in types]


def path_distance(graph, path):
    return sum(graph.edges[a, b]["distance"] for a, b in zip(path, path[1:]))


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def all_agents(state):
    agents = {}
    for kind in ("security", "emergency", "staff", "passenger"):
        agents.update(getattr(state, f"{kind}_locations"))
    return agents


def bfs_route(graph, blocked, start, goal):
    frontier = deque([start])
    parent = {start: None}
    expanded = 0
    while frontier:
        node = frontier.popleft()
        expanded += 1
        if node == goal:
            break
        for neighbor in graph.neighbors(node):
            if neighbor in blocked or neighbor in parent:
                continue
            parent[neighbor] = node
            frontier.append(neighbor)
    if goal not in parent:
        return None, expanded
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1], expanded


def dijkstra_route(graph, state, start, goal):
    tie = count()
    heap = [(0.0, next(tie), start)]
    best = {start: 0.0}
    parent = {}
    closed = set()
    expanded = 0
    while heap:
        cost, _, node = heappop(heap)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1
        if node == goal:
            path = [node]
            while node in parent:
                node = parent[node]
                path.append(node)
            return path[::-1], cost, expanded
        for neighbor in graph.neighbors(node):
            if neighbor in state.blocked_nodes or neighbor in closed:
                continue
            new_cost = cost + edge_cost(node, neighbor, graph, state)
            if new_cost < best.get(neighbor, float("inf")):
                best[neighbor] = new_cost
                parent[neighbor] = node
                heappush(heap, (new_cost, next(tie), neighbor))
    return None, float("inf"), expanded


def exp1_pathfinding(graph, base_state):
    nodes = list(graph.nodes)
    pairs = [(s, g) for s in nodes for g in nodes if s != g]
    totals = {m: {"distance": 0.0, "expanded": 0, "time": 0.0, "optimal": 0} for m in ("BFS (fewest hops)", "Dijkstra", "A*")}

    for start, goal in pairs:
        t0 = perf_counter()
        d_path, d_cost, d_expanded = dijkstra_route(graph, base_state, start, goal)
        d_time = perf_counter() - t0

        t0 = perf_counter()
        b_path, b_expanded = bfs_route(graph, base_state.blocked_nodes, start, goal)
        b_time = perf_counter() - t0
        b_cost = path_distance(graph, b_path)

        t0 = perf_counter()
        a = astar(graph, base_state, start, goal)
        a_time = perf_counter() - t0

        for name, cost, expanded, elapsed in (
            ("BFS (fewest hops)", b_cost, b_expanded, b_time),
            ("Dijkstra", d_cost, d_expanded, d_time),
            ("A*", a.cost, a.nodes_expanded, a_time),
        ):
            t = totals[name]
            t["distance"] += cost
            t["expanded"] += expanded
            t["time"] += elapsed
            t["optimal"] += abs(cost - d_cost) < 1e-6

    rows = []
    for name, t in totals.items():
        n = len(pairs)
        rows.append(
            {
                "method": name,
                "pairs": n,
                "mean_distance": round(t["distance"] / n, 2),
                "mean_nodes_expanded": round(t["expanded"] / n, 2),
                "mean_time_ms": round(t["time"] / n * 1000, 3),
                "optimal_paths_pct": round(t["optimal"] / n * 100, 1),
            }
        )
    return {
        "title": "Experiment 1: Shortest path (BFS) vs Dijkstra vs A*",
        "description": "All 1560 ordered node pairs on the unblocked airport graph.",
        "tables": {"exp1_pathfinding": rows},
    }


def exp2_dynamic_obstacles(graph, base_state):
    agents = all_agents(base_state)
    goals = nodes_of_type(graph, "exit", "terminal")
    blockable = nodes_of_type(graph, "corridor", "checkpoint")

    old_routes = {}
    trips = [(a, s, g) for a, s in agents.items() for g in goals if s != g]
    for agent, start, goal in trips:
        old_routes[(agent, goal)] = astar(graph, base_state, start, goal)

    per_block = []
    totals = {"trips": 0, "affected": 0, "unreachable": 0, "extra": [], "times": [], "replanned_unsafe": 0}
    for block in blockable:
        blocked_state = base_state.block_node(block)
        row = {"blocked_node": block, "trips": 0, "affected": 0, "unreachable": 0, "extra": [], "times": []}
        for agent, start, goal in trips:
            if block in (start, goal):
                continue
            row["trips"] += 1
            old = old_routes[(agent, goal)]
            if block not in old.path:
                continue
            row["affected"] += 1
            t0 = perf_counter()
            new = astar(graph, blocked_state, start, goal)
            row["times"].append((perf_counter() - t0) * 1000)
            if not new.found:
                row["unreachable"] += 1
                continue
            if block in new.path:
                totals["replanned_unsafe"] += 1
            row["extra"].append(new.cost - old.cost)
        per_block.append(row)
        totals["trips"] += row["trips"]
        totals["affected"] += row["affected"]
        totals["unreachable"] += row["unreachable"]
        totals["extra"] += row["extra"]
        totals["times"] += row["times"]

    per_block_rows = [
        {
            "blocked_node": r["blocked_node"],
            "trips": r["trips"],
            "routes_through_block": r["affected"],
            "unreachable_after_block": r["unreachable"],
            "mean_extra_distance": round(mean(r["extra"]), 1) if r["extra"] else "-",
            "mean_replan_ms": round(mean(r["times"]), 3) if r["times"] else "-",
        }
        for r in per_block
    ]
    summary = [
        {
            "strategy": "Static plan (no replanning)",
            "trips": totals["trips"],
            "routes_crossing_blocked_node": totals["affected"],
            "unsafe_routes_pct": round(totals["affected"] / totals["trips"] * 100, 1),
            "rerouted_successfully": "-",
            "unreachable": "-",
            "mean_extra_distance": "-",
            "mean_replan_ms": "-",
        },
        {
            "strategy": "A* with replanning",
            "trips": totals["trips"],
            "routes_crossing_blocked_node": totals["replanned_unsafe"],
            "unsafe_routes_pct": round(totals["replanned_unsafe"] / totals["trips"] * 100, 1),
            "rerouted_successfully": totals["affected"] - totals["unreachable"],
            "unreachable": totals["unreachable"],
            "mean_extra_distance": round(mean(totals["extra"]), 1),
            "mean_replan_ms": round(mean(totals["times"]), 3),
        },
    ]
    return {
        "title": "Experiment 2: A* with vs without dynamic obstacle handling",
        "description": "Each corridor or checkpoint is blocked one at a time. Trips are every agent to every exit and terminal.",
        "tables": {"exp2_dynamic_summary": summary, "exp2_dynamic_per_block": per_block_rows},
    }


def csp_repair_plan(graph, state, agent, start, goal, max_iterations=8):
    extra_blocked = set()
    for _ in range(max_iterations):
        planning_state = replace(state, blocked_nodes=set(state.blocked_nodes) | extra_blocked)
        route = astar(graph, planning_state, start, goal)
        if not route.found:
            return None
        if not validate_route(graph, state, agent, route.path):
            return route.path
        bad = [n for n in route.path if validate_route(graph, state, agent, [n])]
        fixable = [n for n in bad if n not in (start, goal)]
        if not fixable:
            return None
        extra_blocked |= set(fixable)
    return None


def plan_bfs(graph, state, agent, start, goal):
    path, _ = bfs_route(graph, state.blocked_nodes, start, goal)
    return path


def plan_astar(graph, state, agent, start, goal):
    route = astar(graph, state, start, goal)
    return route.path if route.found else None


def make_stress_state(graph, base_state, rng):
    corridors = nodes_of_type(graph, "corridor")
    blocked = set(rng.sample(corridors, rng.choice([0, 1, 2])))
    passenger_homes = set(base_state.passenger_locations.values())
    pool = [
        n for n in nodes_of_type(graph, "corridor", "checkpoint", "terminal")
        if n not in passenger_homes and n not in blocked
    ]
    full = rng.sample(pool, rng.choice([1, 2]))
    occupancy = {n: graph.nodes[n]["capacity"] for n in full}
    return replace(base_state, blocked_nodes=blocked, occupancy=occupancy)


def exp3_csp(graph, base_state):
    strategies = {
        "Baseline BFS": plan_bfs,
        "A* only": plan_astar,
        "A* + CSP": csp_repair_plan,
    }
    stats = {
        name: {"delivered": 0, "rejected": 0, "bad_routes": 0, "violations": 0, "restricted": 0, "capacity": 0, "distance": [], "times": []}
        for name in strategies
    }
    paired = {name: [] for name in strategies}
    passengers = sorted(base_state.passenger_locations)
    total_requests = 0

    for trial in range(CSP_TRIALS):
        rng = random.Random(1000 + trial)
        state = make_stress_state(graph, base_state, rng)
        for _ in range(CSP_REQUESTS_PER_TRIAL):
            agent = rng.choice(passengers)
            start = state.passenger_locations[agent]
            goal = rng.choice([n for n in graph.nodes if n != start and n not in state.blocked_nodes])
            total_requests += 1
            distances = {}
            for name, planner in strategies.items():
                t0 = perf_counter()
                path = planner(graph, state, agent, start, goal)
                elapsed = (perf_counter() - t0) * 1000
                s = stats[name]
                s["times"].append(elapsed)
                if path is None:
                    s["rejected"] += 1
                    distances[name] = None
                    continue
                s["delivered"] += 1
                violations = validate_route(graph, state, agent, path)
                if violations:
                    s["bad_routes"] += 1
                s["violations"] += len(violations)
                s["restricted"] += sum("restricted zone" in v for v in violations)
                s["capacity"] += sum("full capacity" in v for v in violations)
                distance = path_distance(graph, path)
                s["distance"].append(distance)
                distances[name] = distance
            if all(d is not None for d in distances.values()):
                for name, d in distances.items():
                    paired[name].append(d)

    rows = []
    for name, s in stats.items():
        rows.append(
            {
                "strategy": name,
                "requests": total_requests,
                "routes_delivered": s["delivered"],
                "requests_rejected": s["rejected"],
                "routes_with_violation": s["bad_routes"],
                "compliant_routes": s["delivered"] - s["bad_routes"],
                "violation_rate_pct": round(s["bad_routes"] / max(s["delivered"], 1) * 100, 1),
                "restricted_zone_violations": s["restricted"],
                "capacity_violations": s["capacity"],
                "mean_distance_delivered": round(mean(s["distance"]), 1),
                "mean_distance_paired": round(mean(paired[name]), 1),
                "mean_plan_ms": round(mean(s["times"]), 3),
            }
        )
    return {
        "title": "Experiment 3: Baseline vs A* vs A* + CSP (ablation: remove CSP)",
        "description": (
            f"{CSP_TRIALS} random stress states (0-2 blocked corridors, 1-2 full nodes) with "
            f"{CSP_REQUESTS_PER_TRIAL} passenger requests each. Violations are counted by the CSP validator."
        ),
        "tables": {"exp3_csp": rows},
    }


def random_incidents(graph, rng):
    while True:
        n_sec, n_em, n_staff = rng.choice([1, 2, 3]), rng.choice([0, 1, 2]), rng.choice([0, 1])
        if n_sec + n_em + n_staff >= 2:
            break
    kinds = (
        [rng.choice(["threat", "unauthorized_access"]) for _ in range(n_sec)]
        + [rng.choice(["fire", "medical"]) for _ in range(n_em)]
        + ["corridor_blocked"] * n_staff
    )
    sites = nodes_of_type(graph, "gate", "terminal", "corridor", "restricted", "staff_area", "checkpoint")
    return [
        Incident(f"INC_{i + 1}", kind, rng.choice(sites), rng.choice(["high", "high", "medium", "low"]))
        for i, kind in enumerate(kinds)
    ]


def exp4_ga(graph, base_state):
    names = ["Random assignment", "Greedy (A* + CSP)", "GA (A* + CSP + GA)", "Brute-force optimum"]
    stats = {n: {"response": [], "fitness": [], "violations": 0, "bad": 0, "gap": [], "optimal": 0, "times": []} for n in names}
    versus = {"better": 0, "tie": 0, "worse": 0}
    convergence = []
    generations = GAConfig().generations

    for trial in range(GA_TRIALS):
        rng = random.Random(5000 + trial)
        incidents = random_incidents(graph, rng)
        state = base_state
        for incident in incidents:
            state = state.add_incident(incident)
        cost_fn, _ = make_cost_fn(graph, state)
        domains = build_domains(state, incidents)

        def response_cost(assignment):
            return sum(SEVERITY_WEIGHT[i.severity] * cost_fn(a, i.location_node) for a, i in assignment)

        t0 = perf_counter()
        random_chromosome = [rng.choice(d) for d in domains]
        random_assignment = list(zip(random_chromosome, incidents))
        random_time = (perf_counter() - t0) * 1000

        t0 = perf_counter()
        greedy_assign, _ = greedy_assignment(graph, state, incidents)
        greedy_time = (perf_counter() - t0) * 1000

        ga = run_ga(graph, state, incidents=incidents, config=GAConfig(seed=trial))

        t0 = perf_counter()
        optimal_assign, optimal_fitness = brute_force_best(graph, state, incidents)
        optimal_time = (perf_counter() - t0) * 1000

        outcomes = {
            names[0]: (random_assignment, random_time),
            names[1]: (greedy_assign, greedy_time),
            names[2]: (ga.assignment, ga.time_ms),
            names[3]: (optimal_assign, optimal_time),
        }
        fitnesses = {}
        for name, (assignment, elapsed) in outcomes.items():
            chromosome = [a for a, _ in assignment]
            fit = fitness(chromosome, incidents, state, cost_fn)
            violations = len(validate_assignment(state, assignment))
            fitnesses[name] = fit
            s = stats[name]
            s["response"].append(response_cost(assignment))
            s["fitness"].append(fit)
            s["violations"] += violations
            s["bad"] += violations > 0
            s["gap"].append((fit - optimal_fitness) / optimal_fitness * 100)
            s["optimal"] += abs(fit - optimal_fitness) < 1e-6
            s["times"].append(elapsed)

        diff = fitnesses[names[2]] - fitnesses[names[1]]
        versus["better" if diff < -1e-6 else "worse" if diff > 1e-6 else "tie"] += 1
        convergence.append([(h - optimal_fitness) / optimal_fitness * 100 for h in ga.best_history])

    rows = []
    for name, s in stats.items():
        rows.append(
            {
                "strategy": name,
                "trials": GA_TRIALS,
                "mean_response_cost": round(mean(s["response"]), 1),
                "mean_fitness": round(mean(s["fitness"]), 1),
                "trials_with_violation": s["bad"],
                "total_violations": s["violations"],
                "optimal_pct": round(s["optimal"] / GA_TRIALS * 100, 1),
                "mean_gap_to_optimal_pct": round(mean(s["gap"]), 2),
                "mean_time_ms": round(mean(s["times"]), 2),
            }
        )
    curve = [
        {"generation": g + 1, "mean_gap_pct": round(mean(c[g] for c in convergence), 3)}
        for g in range(generations)
    ]
    versus_rows = [
        {"comparison": "GA vs greedy", "GA_better": versus["better"], "tie": versus["tie"], "GA_worse": versus["worse"]}
    ]
    return {
        "title": "Experiment 4: A* + CSP vs A* + CSP + GA (ablation: remove GA)",
        "description": (
            f"{GA_TRIALS} random incident sets (2-6 incidents). Fitness = severity-weighted response distance "
            "plus a large penalty per CSP violation. Optimum found by brute force."
        ),
        "tables": {"exp4_ga": rows, "exp4_ga_vs_greedy": versus_rows, "exp4_ga_convergence": curve},
    }


def exp4b_ga_scaling(graph, base_state):
    trials = 30
    n = 8
    config = GAConfig(population_size=60, generations=80)
    sites = nodes_of_type(graph, "gate", "terminal", "corridor", "restricted", "checkpoint")
    homes = nodes_of_type(graph, "checkpoint", "corridor", "terminal")
    greedy_fit, ga_fit, greedy_ms, ga_ms = [], [], [], []
    versus = {"better": 0, "tie": 0, "worse": 0}
    violations = 0
    for trial in range(trials):
        rng = random.Random(7000 + trial)
        state = base_state
        for i in range(n - len(state.security_locations)):
            state = state.move_agent(f"SEC_X{i + 1}", "security", rng.choice(homes))
        incidents = [
            Incident(f"INC_{i + 1}", "threat", rng.choice(sites), rng.choice(["high", "high", "medium", "low"]))
            for i in range(n)
        ]
        for incident in incidents:
            state = state.add_incident(incident)

        t0 = perf_counter()
        greedy_assign, g_fit = greedy_assignment(graph, state, incidents)
        greedy_ms.append((perf_counter() - t0) * 1000)

        ga = run_ga(graph, state, incidents=incidents, config=GAConfig(population_size=config.population_size, generations=config.generations, seed=trial))
        ga_ms.append(ga.time_ms)
        violations += len(validate_assignment(state, ga.assignment))

        greedy_fit.append(g_fit)
        ga_fit.append(ga.best_fitness)
        diff = ga.best_fitness - g_fit
        versus["better" if diff < -1e-6 else "worse" if diff > 1e-6 else "tie"] += 1

    rows = [
        {
            "strategy": "Greedy (A* + CSP)",
            "mean_fitness": round(mean(greedy_fit), 1),
            "mean_time_ms": round(mean(greedy_ms), 2),
        },
        {
            "strategy": "GA (population 60, 80 generations)",
            "mean_fitness": round(mean(ga_fit), 1),
            "mean_time_ms": round(mean(ga_ms), 2),
        },
    ]
    rows.append(
        {
            "strategy": "GA improvement over greedy",
            "mean_fitness": f"{round((mean(greedy_fit) - mean(ga_fit)) / mean(greedy_fit) * 100, 1)}%",
            "mean_time_ms": f"better {versus['better']}, tie {versus['tie']}, worse {versus['worse']} of {trials}",
        }
    )
    return {
        "title": "Experiment 4b: GA at larger scale (8 agents, 8 incidents)",
        "description": (
            f"{trials} random instances. Brute force is infeasible here ({n}^{n} = {n ** n:,} assignments), "
            f"so GA is compared with greedy. GA solutions had {violations} CSP violations in total."
        ),
        "tables": {"exp4b_ga_scaling": rows},
    }


def exp5_minimax(graph, base_state):
    exits = set(nodes_of_type(graph, "exit"))
    starts = sorted(n for n in graph.nodes if n not in base_state.restricted_zones and n not in exits)
    rows = []
    for attacker in ("greedy", "minimax"):
        for defender in ("idle", "chase", "minimax"):
            runs = [simulate(graph, base_state, defender, attacker, threat_start=s) for s in starts]
            n = len(runs)
            rows.append(
                {
                    "attacker": attacker,
                    "defender": defender if defender != "minimax" else "minimax (with)",
                    "threat_starts": n,
                    "intercepted_pct": round(sum(r.outcome == "intercepted" for r in runs) / n * 100, 1),
                    "breach_pct": round(sum(r.outcome == "breach" for r in runs) / n * 100, 1),
                    "contained_pct": round(sum(r.outcome == "contained" for r in runs) / n * 100, 1),
                    "mean_peak_risk": round(mean(r.peak_risk for r in runs), 1),
                    "mean_rounds": round(mean(r.rounds for r in runs), 2),
                    "mean_search_nodes_per_run": round(mean(r.nodes_evaluated for r in runs), 1) if defender == "minimax" or attacker == "minimax" else "-",
                }
            )

    with_nodes, without_nodes, same = [], [], 0
    officers = tuple(sorted(base_state.security_locations.items()))
    for start in starts:
        gs = GameState(start, officers, frozenset())
        ctx_a = GameContext(graph, base_state, default_protected(graph, base_state, start))
        ctx_b = GameContext(graph, base_state, default_protected(graph, base_state, start))
        a = best_max_action(gs, 4, ctx_a, pruning=True)
        b = best_max_action(gs, 4, ctx_b, pruning=False)
        with_nodes.append(ctx_a.nodes_evaluated)
        without_nodes.append(ctx_b.nodes_evaluated)
        same += a == b
    pruning_rows = [
        {
            "search": "Minimax without pruning",
            "mean_nodes_evaluated": round(mean(without_nodes), 1),
            "same_decision_pct": 100.0,
        },
        {
            "search": "Minimax with alpha-beta",
            "mean_nodes_evaluated": round(mean(with_nodes), 1),
            "same_decision_pct": round(same / len(starts) * 100, 1),
        },
    ]
    return {
        "title": "Experiment 5: Security response with vs without minimax (ablation: remove minimax)",
        "description": "Threat starts at each of 33 non-protected nodes. Depth 4 for the defender, depth 3 for the attacker.",
        "tables": {"exp5_minimax": rows, "exp5_pruning": pruning_rows},
    }


def oracle_actions(graph, state):
    actions = set()
    types = {i.type for i in state.active_incidents}
    if "fire" in types:
        actions.add("activate_evacuation")
    if state.threat_location is not None or "threat" in types:
        actions.add("dispatch_security")
    if any(graph.nodes[n]["type"] == "corridor" for n in state.blocked_nodes if n in graph.nodes):
        actions.add("recalculate_routes")
    if "unauthorized_access" in types:
        actions.add("trigger_security_alert")
    if any(i.severity == "high" for i in state.active_incidents):
        actions.add("prioritize_emergency_response")
    if any(count > 0.8 * graph.nodes[n]["capacity"] for n, count in state.occupancy.items() if n in graph.nodes):
        actions.add("apply_congestion_penalty")
    return actions


def random_expert_state(graph, base_state, rng):
    state = base_state
    kinds = ["fire", "threat", "unauthorized_access", "corridor_blocked", "medical"]
    sites = nodes_of_type(graph, "gate", "terminal", "corridor", "restricted")
    for i, kind in enumerate(rng.sample(kinds, rng.choice([0, 1, 2, 3]))):
        state = state.add_incident(Incident(f"INC_{i + 1}", kind, rng.choice(sites), rng.choice(["high", "medium", "low"])))
    if rng.random() < 0.3:
        state = state.set_threat_location(rng.choice(sites))
    for node in rng.sample(nodes_of_type(graph, "corridor", "gate"), rng.choice([0, 1, 2])):
        state = state.block_node(node)
    crowded = rng.sample(nodes_of_type(graph, "terminal", "corridor"), rng.choice([0, 1]))
    return replace(state, occupancy={n: int(graph.nodes[n]["capacity"] * rng.choice([0.5, 0.9])) for n in crowded})


def prf(triggered, needed):
    tp = len(triggered & needed)
    fp = len(triggered - needed)
    fn = len(needed - triggered)
    return tp, fp, fn


def exp6_expert_system(graph, base_state):
    expected = {
        "scenario_1_normal": set(),
        "scenario_2_blocked_corridor": {"recalculate_routes"},
        "scenario_3_security_threat": {"dispatch_security", "prioritize_emergency_response"},
        "scenario_4_evacuation": {"activate_evacuation", "prioritize_emergency_response"},
        "scenario_5_unauthorized_access": {"trigger_security_alert"},
    }
    scenario_rows = []
    correct = 0
    for name, want in expected.items():
        state = build_initial_state(graph, name)
        got = {f["action"] for f in run_expert_system(graph, state)}
        repeat = {f["action"] for f in run_expert_system(graph, state)}
        correct += got == want
        scenario_rows.append(
            {
                "scenario": name,
                "expected_actions": ", ".join(sorted(want)) or "none",
                "fired_actions": ", ".join(sorted(got)) or "none",
                "match": got == want,
                "repeat_run_identical": got == repeat,
            }
        )

    rng = random.Random(9000)
    trials = 300
    agree, deterministic = 0, 0
    pooled = {"Expert system": [0, 0, 0], "No expert system (nothing triggered)": [0, 0, 0], "No expert system (trigger everything)": [0, 0, 0]}
    for _ in range(trials):
        state = random_expert_state(graph, base_state, rng)
        needed = oracle_actions(graph, state)
        fired = {f["action"] for f in run_expert_system(graph, state)}
        agree += fired == needed
        deterministic += fired == {f["action"] for f in run_expert_system(graph, state)}
        for name, triggered in (
            ("Expert system", fired),
            ("No expert system (nothing triggered)", set()),
            ("No expert system (trigger everything)", set(ALL_ACTIONS)),
        ):
            for i, v in enumerate(prf(triggered, needed)):
                pooled[name][i] += v

    ablation_rows = []
    for name, (tp, fp, fn) in pooled.items():
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ablation_rows.append(
            {
                "configuration": name,
                "correct_actions": tp,
                "unneeded_actions": fp,
                "missed_actions": fn,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
            }
        )
    summary_rows = [
        {"test": "Scenario decisions match the specification", "result": f"{correct}/{len(expected)}"},
        {"test": f"Random states where rule output equals independent specification ({trials})", "result": f"{agree}/{trials}"},
        {"test": f"Random states with identical output on repeat run ({trials})", "result": f"{deterministic}/{trials}"},
    ]
    return {
        "title": "Experiment 6: Expert system decision accuracy, consistency and ablation",
        "description": "The specification is written independently of the rule engine. The ablation compares the actions triggered without an expert system.",
        "tables": {"exp6_summary": summary_rows, "exp6_scenarios": scenario_rows, "exp6_ablation": ablation_rows},
    }


def exp7_integrated(graph, base_state):
    rows = []
    for name in load_scenarios():
        state = build_initial_state(graph, name)
        times = []
        result = None
        for _ in range(10):
            t0 = perf_counter()
            result = run_simulation(graph, state, name)
            times.append((perf_counter() - t0) * 1000)
        rows.append(
            {
                "scenario": name,
                "rules_fired": len(result.expert_system_trace),
                "routes_planned": len(result.route_table),
                "responders_assigned": len(result.ga_assignment["rows"]) if result.ga_assignment else 0,
                "csp_violations": len(result.csp_report["violations"]),
                "access_attempts_denied": sum(not a["allowed"] for a in result.csp_report["access_denied"]),
                "threat_outcome": result.minimax_outcome["outcome"] if result.minimax_outcome else "-",
                "mean_total_time_ms": round(mean(times), 1),
            }
        )
    return {
        "title": "Experiment 7: Integrated simulation across all scenarios",
        "description": "Full pipeline (expert system, A*, CSP, GA, minimax) run 10 times per scenario.",
        "tables": {"exp7_integrated": rows},
    }


def md_table(rows):
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def save_csv(name, rows):
    with open(os.path.join(RESULTS_DIR, f"{name}.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_all(results):
    tables = {}
    for r in results:
        tables.update(r["tables"])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    rows = tables["exp1_pathfinding"]
    ax.bar([r["method"] for r in rows], [r["mean_nodes_expanded"] for r in rows], color=["#94A3B8", "#F59E0B", "#22C55E"])
    ax.set_ylabel("mean nodes expanded per query")
    ax.set_title("Search effort (all node pairs)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig1_search_effort.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    rows = tables["exp3_csp"]
    ax.bar([r["strategy"] for r in rows], [r["violation_rate_pct"] for r in rows], color=["#94A3B8", "#F59E0B", "#22C55E"])
    ax.set_ylabel("routes with a constraint violation (%)")
    ax.set_title("Constraint violations by planner")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig2_csp_violations.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    rows = tables["exp4_ga_convergence"]
    ax.plot([r["generation"] for r in rows], [r["mean_gap_pct"] for r in rows], color="#22C55E", linewidth=2)
    ax.set_xlabel("generation")
    ax.set_ylabel("mean gap to optimum (%)")
    ax.set_title("GA convergence (best individual)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig3_ga_convergence.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    rows = [r for r in tables["exp5_minimax"] if r["attacker"] == "minimax"]
    labels = [r["defender"] for r in rows]
    bottom = [0] * len(rows)
    for key, color in (("intercepted_pct", "#22C55E"), ("contained_pct", "#F59E0B"), ("breach_pct", "#EF4444")):
        values = [r[key] for r in rows]
        ax.bar(labels, values, bottom=bottom, label=key.replace("_pct", ""), color=color)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_ylabel("% of threat starts")
    ax.set_title("Outcome against a minimax attacker")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig4_minimax_outcomes.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    graph = load_airport_graph()
    base_state = build_initial_state(graph, "scenario_1_normal")

    runners = [
        exp1_pathfinding,
        exp2_dynamic_obstacles,
        exp3_csp,
        exp4_ga,
        exp4b_ga_scaling,
        exp5_minimax,
        exp6_expert_system,
        exp7_integrated,
    ]
    results = []
    markdown = ["# Experiment results", "", "Generated by `python -m evaluation.experiments`. All numbers come from running the code.", ""]
    for runner in runners:
        t0 = perf_counter()
        result = runner(graph, base_state)
        elapsed = perf_counter() - t0
        results.append(result)
        print(f"\n## {result['title']}  ({elapsed:.1f}s)")
        print(result["description"])
        markdown += [f"## {result['title']}", "", result["description"], ""]
        for table_name, rows in result["tables"].items():
            save_csv(table_name, rows)
            if table_name == "exp4_ga_convergence":
                shown = [rows[0], rows[len(rows) // 2 - 1], rows[-1]]
                print(f"\n{table_name} (first, middle, last generation)")
                print(md_table(shown))
                markdown += [f"**{table_name}** (first, middle, last generation; full data in CSV)", "", md_table(shown), ""]
                continue
            print(f"\n{table_name}")
            print(md_table(rows))
            markdown += [f"**{table_name}**", "", md_table(rows), ""]

    plot_all(results)
    with open(os.path.join(RESULTS_DIR, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(markdown))
    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump({r["title"]: r["tables"] for r in results}, f, indent=2)
    print(f"\nSaved tables, figures and results.md to {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()