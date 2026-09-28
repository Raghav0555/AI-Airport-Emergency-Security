import random
from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import List, Tuple

from src.astar.astar import astar
from src.csp.constraints import REQUIRED_AGENT_TYPE, validate_assignment

SEVERITY_WEIGHT = {"high": 3.0, "medium": 2.0, "low": 1.0}
VIOLATION_PENALTY = 100000.0
UNREACHABLE_COST = 5000.0
AGENT_KINDS = ("security", "emergency", "staff", "passenger")


@dataclass
class GAConfig:
    population_size: int = 30
    generations: int = 40
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    tournament_size: int = 3
    elitism: int = 2
    seed: int = 42


@dataclass
class GAResult:
    assignment: List[Tuple[str, object]]
    best_fitness: float
    best_history: List[float]
    avg_history: List[float]
    astar_calls: int
    time_ms: float


def agent_location(state, agent_id):
    for kind in AGENT_KINDS:
        locations = getattr(state, f"{kind}_locations")
        if agent_id in locations:
            return locations[agent_id]
    raise KeyError(agent_id)


def build_domains(state, incidents):
    domains = []
    for incident in incidents:
        kind = REQUIRED_AGENT_TYPE.get(incident.type)
        if kind is None:
            raise ValueError(f"No agent type defined for incident type '{incident.type}'")
        domains.append(list(getattr(state, f"{kind}_locations")))
    return domains


def make_cost_fn(graph, state):
    cache = {}
    counter = {"calls": 0}

    def cost(agent_id, node):
        key = (agent_id, node)
        if key not in cache:
            result = astar(graph, state, agent_location(state, agent_id), node)
            counter["calls"] += 1
            cache[key] = result.cost if result.found else UNREACHABLE_COST
        return cache[key]

    return cost, counter


def fitness(chromosome, incidents, state, cost_fn):
    assignment = list(zip(chromosome, incidents))
    response_cost = sum(
        SEVERITY_WEIGHT[incident.severity] * cost_fn(agent_id, incident.location_node)
        for agent_id, incident in assignment
    )
    violations = len(validate_assignment(state, assignment))
    return response_cost + VIOLATION_PENALTY * violations


def tournament(population, scores, rng, size):
    contenders = rng.sample(range(len(population)), size)
    winner = min(contenders, key=lambda i: scores[i])
    return population[winner]


def crossover(parent_a, parent_b, rng):
    return [a if rng.random() < 0.5 else b for a, b in zip(parent_a, parent_b)]


def mutate(chromosome, domains, rng, rate):
    return [
        rng.choice(domains[i]) if rng.random() < rate else gene
        for i, gene in enumerate(chromosome)
    ]


def run_ga(graph, state, incidents=None, config=None):
    t0 = perf_counter()
    config = config or GAConfig()
    incidents = incidents if incidents is not None else state.active_incidents
    rng = random.Random(config.seed)

    domains = build_domains(state, incidents)
    cost_fn, counter = make_cost_fn(graph, state)

    population = [[rng.choice(d) for d in domains] for _ in range(config.population_size)]
    best_history, avg_history = [], []
    best_chromosome, best_score = None, float("inf")

    for _ in range(config.generations):
        scores = [fitness(c, incidents, state, cost_fn) for c in population]

        generation_best = min(range(len(population)), key=lambda i: scores[i])
        if scores[generation_best] < best_score:
            best_score = scores[generation_best]
            best_chromosome = list(population[generation_best])
        best_history.append(best_score)
        avg_history.append(sum(scores) / len(scores))

        ranked = sorted(range(len(population)), key=lambda i: scores[i])
        next_population = [list(population[i]) for i in ranked[: config.elitism]]

        while len(next_population) < config.population_size:
            parent_a = tournament(population, scores, rng, config.tournament_size)
            parent_b = tournament(population, scores, rng, config.tournament_size)
            if rng.random() < config.crossover_rate:
                child = crossover(parent_a, parent_b, rng)
            else:
                child = list(parent_a)
            next_population.append(mutate(child, domains, rng, config.mutation_rate))

        population = next_population

    assignment = list(zip(best_chromosome, incidents))
    return GAResult(
        assignment=assignment,
        best_fitness=best_score,
        best_history=best_history,
        avg_history=avg_history,
        astar_calls=counter["calls"],
        time_ms=(perf_counter() - t0) * 1000,
    )


def greedy_assignment(graph, state, incidents=None):
    incidents = incidents if incidents is not None else state.active_incidents
    cost_fn, _ = make_cost_fn(graph, state)
    domains = build_domains(state, incidents)
    order = sorted(range(len(incidents)), key=lambda i: -SEVERITY_WEIGHT[incidents[i].severity])

    chromosome = [None] * len(incidents)
    used = set()
    for i in order:
        available = [a for a in domains[i] if a not in used] or domains[i]
        chosen = min(available, key=lambda a: cost_fn(a, incidents[i].location_node))
        chromosome[i] = chosen
        used.add(chosen)

    score = fitness(chromosome, incidents, state, cost_fn)
    return list(zip(chromosome, incidents)), score


def brute_force_best(graph, state, incidents=None):
    incidents = incidents if incidents is not None else state.active_incidents
    cost_fn, _ = make_cost_fn(graph, state)
    domains = build_domains(state, incidents)

    best_chromosome, best_score = None, float("inf")
    for chromosome in product(*domains):
        score = fitness(list(chromosome), incidents, state, cost_fn)
        if score < best_score:
            best_chromosome, best_score = list(chromosome), score
    return list(zip(best_chromosome, incidents)), best_score


if __name__ == "__main__":
    from src.csp.constraints import validate_assignment
    from src.world.graph_loader import load_airport_graph
    from src.world.scenario_loader import build_initial_state
    from src.world.state import Incident

    g = load_airport_graph()
    state = build_initial_state(g, "scenario_1_normal")
    for incident in [
        Incident("INC_A", "threat", "RESTRICTED_1", "high"),
        Incident("INC_B", "unauthorized_access", "RESTRICTED_2", "medium"),
        Incident("INC_C", "threat", "G3_2", "high"),
        Incident("INC_D", "fire", "G1_2", "high"),
        Incident("INC_E", "medical", "G2_4", "medium"),
    ]:
        state = state.add_incident(incident)

    result = run_ga(g, state)
    greedy_assignment_result, greedy_score = greedy_assignment(g, state)
    optimal_assignment, optimal_score = brute_force_best(g, state)

    print("GA assignment:")
    for agent_id, incident in result.assignment:
        print(f"  {incident.incident_id} ({incident.type}, {incident.severity}) <- {agent_id}")
    print("GA best fitness:", round(result.best_fitness, 2))
    print("greedy fitness:", round(greedy_score, 2))
    print("brute-force optimal fitness:", round(optimal_score, 2))
    print("GA matches optimal:", abs(result.best_fitness - optimal_score) < 1e-6)
    print("generation 1 best/avg:", round(result.best_history[0], 2), "/", round(result.avg_history[0], 2))
    print("final best/avg:", round(result.best_history[-1], 2), "/", round(result.avg_history[-1], 2))
    print("GA violations:", len(validate_assignment(state, result.assignment)))
    print("greedy violations:", len(validate_assignment(state, greedy_assignment_result)))
    print("A* calls:", result.astar_calls, "| time ms:", round(result.time_ms, 1))