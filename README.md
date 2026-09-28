# AI Airport Emergency & Security Management Simulator

An academic AI simulation prototype for the AIES course — a mini expert system for a real-world domain (airport emergency and security management), built using state-space representation, A*, a genetic algorithm, CSP, and minimax. **Not** a real-world deployment system.

Full methodology, architecture, and results: [`docs/report.md`](docs/report.md)

## Key results

| Technique | Result |
|---|---|
| A* vs Dijkstra vs BFS | 100% optimal paths, ~half the nodes expanded vs Dijkstra |
| A* + CSP vs baseline | 0% constraint violations vs ~50% for BFS/A* alone |
| GA vs greedy (8 agents/incidents) | 3.6% better fitness, where brute force is infeasible |
| Minimax vs simple chase defense | 100% threat interception vs 90.9% |
| Expert system vs specification | 300/300 random states, 5/5 scenarios match |

Full experiment tables, CSVs and plots: [`docs/results/`](docs/results/)

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m evaluation.experiments           # run all experiments
python -m streamlit run app/streamlit_app.py   # interactive dashboard
```

## Project structure

```
data/            synthetic airport graph, agents, scenarios
src/              A*, CSP, genetic algorithm, minimax, expert system, chatbot
simulation/        shared-state orchestrator
evaluation/         experiments and ablation studies
docs/                report, architecture docs, results
app/                  Streamlit dashboard
```