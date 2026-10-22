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


## CI/CD Pipeline (CCD CCA 2)

This project is the AI Airport Emergency & Security Management Simulator, a Streamlit application built for the AIES course. It is deployed through a Jenkins CI/CD pipeline running on AWS, with Docker for packaging and Kubernetes (Minikube) for deployment.

### Architecture

GitHub -> Jenkins (AWS EC2) -> Docker image -> Docker Hub -> Minikube (Kubernetes) -> Streamlit app on NodePort 30080

### Pipeline stages (Jenkinsfile)

1. **Checkout**: Jenkins pulls the latest code from this GitHub repository.
2. **Build & Test**: runs a Python syntax check (`python3 -m compileall`) over all source folders, since the project has no unit-test suite.
3. **Docker Build**: builds the image from the `Dockerfile` and tags it with the Jenkins build number and `latest`.
4. **Push Image**: logs in to Docker Hub using credentials stored securely in Jenkins (`dockerhub-creds`) and pushes both tags to `raghav1089/aies-app`.
5. **Minikube Deployment**: applies the manifests in `k8s/` and updates the Deployment to the new image tag, then waits for the rollout to finish.
6. **Verification**: lists pods, deployments and services, and calls the Streamlit health endpoint (`/_stcore/health`) on the Minikube NodePort.
7. **Post Actions**: prints the result and removes unused Docker images.

### Jenkins setup on AWS

- Launched an EC2 instance (`c7i-flex.large`, Ubuntu 26.04 LTS, 30 GB disk) in ap-south-1 with a security group allowing SSH (22) and Jenkins (8080).
- Installed Docker, OpenJDK 21, Jenkins (LTS), `kubectl` and Minikube on the instance.
- Added the `jenkins` user to the `docker` group and started Minikube as the `jenkins` user with the Docker driver.
- Added Docker Hub credentials in Jenkins (ID `dockerhub-creds`) and created a Pipeline job that reads the `Jenkinsfile` from this repository (Pipeline script from SCM).

### Image registry

The image is pushed to Docker Hub as `raghav1089/aies-app`. Every build creates a tag with the build number (for example `1`, `2`) plus `latest`.

### Deployment and access

- `k8s/deployment.yaml`: one replica, container port 8501, readiness and liveness probes on the Streamlit health endpoint.
- `k8s/services.yaml`: NodePort service exposing the app on port 30080.
- The app is reached from a laptop through an SSH tunnel to the Minikube IP on port 30080, then opened at `http://localhost:30080`.

### Second run (code change)

The sidebar of the app shows `APP_VERSION`. After the first successful run (version 1.0.0), the value in `app/streamlit_app.py` was changed to 2.0.0 and pushed. Jenkins built image tag `2`, pushed it, rolled out the new pod, and the app then showed version 2.0.0.
```