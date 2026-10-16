import itertools
import os
from collections import Counter
from time import perf_counter

import networkx as nx

from evaluation.resilience import (
    RESULTS_DIR,
    candidate_edges,
    greedy_additions,
    md_table,
    save_csv,
    trip_list,
    with_edges,
)
from evaluation.experiments import nodes_of_type
from src.world.graph_loader import load_airport_graph
from src.world.scenario_loader import build_initial_state

MAX_K = 4


def build_tables(graph, trips, blockable, candidates):
    tables = []
    for block in blockable:
        h = graph.copy()
        h.remove_node(block)
        comp = {}
        for i, members in enumerate(nx.connected_components(h)):
            for node in members:
                comp[node] = i
        pairs = Counter()
        for _, start, goal in trips:
            if block in (start, goal):
                continue
            if comp[start] != comp[goal]:
                pairs[(comp[start], comp[goal])] += 1
        merges = []
        for u, v, _ in candidates:
            if block in (u, v) or comp[u] == comp[v]:
                merges.append(None)
            else:
                merges.append((comp[u], comp[v]))
        tables.append((list(pairs.items()), merges, len(set(comp.values()))))
    return tables


def lost_trips(tables, chosen):
    total = 0
    for pairs, merges, n_comp in tables:
        active = [merges[i] for i in chosen if merges[i] is not None]
        if not active:
            total += sum(count for _, count in pairs)
            continue
        parent = list(range(n_comp))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in active:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        for (cs, cg), count in pairs:
            if find(cs) != find(cg):
                total += count
    return total


def main():
    graph = load_airport_graph()
    base_state = build_initial_state(graph, "scenario_1_normal")
    blockable = nodes_of_type(graph, "corridor", "checkpoint")
    trips = trip_list(graph, base_state)
    candidates = candidate_edges(graph)
    tables = build_tables(graph, trips, blockable, candidates)
    denom = sum(1 for b in blockable for _, s, g in trips if b not in (s, g))

    greedy_steps = {k: lost for k, _, lost in greedy_additions(graph, trips, blockable, candidates, MAX_K)}
    floor = lost_trips(tables, range(len(candidates)))

    rows = []
    for k in range(MAX_K + 1):
        t0 = perf_counter()
        best = None
        best_set = None
        count = 0
        for chosen in itertools.combinations(range(len(candidates)), k):
            value = lost_trips(tables, chosen)
            count += 1
            if best is None or value < best:
                best = value
                best_set = chosen
        greedy_value = greedy_steps.get(k, greedy_steps[max(greedy_steps)])
        rows.append(
            {
                "edges_added": k,
                "link_sets_checked": count,
                "exhaustive_best_lost": best,
                "exhaustive_best_pct": round(best / denom * 100, 1),
                "greedy_lost": greedy_value,
                "greedy_pct": round(greedy_value / denom * 100, 1),
                "greedy_extra_lost": greedy_value - best,
                "best_set": "; ".join(f"{candidates[i][0]}-{candidates[i][1]}" for i in best_set),
                "seconds": round(perf_counter() - t0, 1),
            }
        )
        print(
            f"K={k}: exhaustive {best}, greedy {greedy_value}, sets {count}, "
            f"{rows[-1]['seconds']}s"
        )

    print(f"Lost trips with every candidate link added: {floor}")
    save_csv("exp8_exact_vs_greedy", rows)
    lines = [
        "# Experiment 8b: Greedy link choice vs exhaustive search",
        "",
        f"Every possible set of up to {MAX_K} links was checked against {len(candidates)} candidates.",
        f"Lost trips with all {len(candidates)} candidates added at once: {floor} of {denom}.",
        "",
        md_table(rows),
        "",
    ]
    with open(os.path.join(RESULTS_DIR, "resilience_exact.md"), "w") as f:
        f.write("\n".join(lines))
    print("saved docs/results/resilience_exact.md")


if __name__ == "__main__":
    main()