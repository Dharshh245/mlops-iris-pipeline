# MLOps Iris Pipeline — Reproducible experiments to Kubernetes

One repository answering both assignment questions. Question 1 builds a reproducible
training pipeline (Git + DVC + CML); Question 2 serves the model that pipeline produces
as a scaled service on Kubernetes with CI/CD.

| Part | Where |
|---|---|
| Q1 (a) Notebook: accuracy + F1 | [`notebooks/01_iris_experiment.ipynb`](notebooks/01_iris_experiment.ipynb) |
| Q1 (b) Scripts + params | [`src/prepare.py`](src/prepare.py), [`src/train.py`](src/train.py), [`src/evaluate.py`](src/evaluate.py), [`params.yaml`](params.yaml) |
| Q1 (c) Git + DVC + remote | [`scripts/q1_run_experiment.sh`](scripts/q1_run_experiment.sh) (steps 1–3) |
| Q1 (d) Pipeline + diffs | [`dvc.yaml`](dvc.yaml), same script (steps 4–6) |
| Q1 (e) CML on pull requests | [`.github/workflows/cml.yaml`](.github/workflows/cml.yaml) |
| Q2 (a) API + Dockerfile | [`api/main.py`](api/main.py), [`Dockerfile`](Dockerfile) |
| Q2 (b) Kubernetes + scaling | [`k8s/`](k8s), [`scripts/q2_deploy_local.sh`](scripts/q2_deploy_local.sh) |
| Q2 (c) MLflow + Docker CI, SHA tags, registry | [`src/package_mlflow.py`](src/package_mlflow.py), [`.github/workflows/cicd.yaml`](.github/workflows/cicd.yaml) (job `build`) |
| Q2 (d) CD with rollout verification + rollback | [`.github/workflows/cicd.yaml`](.github/workflows/cicd.yaml) (job `deploy`) |
| Q2 (e) Pipeline diagram | [`docs/pipeline.md`](docs/pipeline.md) |

```
├── notebooks/01_iris_experiment.ipynb
├── src/            get_data.py · prepare.py · train.py · evaluate.py · package_mlflow.py
├── api/            main.py (FastAPI) · model_service.py · requirements.txt
├── k8s/            deployment.yaml · service.yaml · hpa.yaml
├── tests/          test_model_service.py · test_api.py
├── scripts/        q1_run_experiment.sh · q2_deploy_local.sh
├── .github/workflows/  cml.yaml (Q1e) · cicd.yaml (Q2c, Q2d)
├── params.yaml · dvc.yaml · requirements.txt · Dockerfile
└── docs/pipeline.md
```

---

## Question 1 — Reproducible ML experiment with Git, DVC and CML

### (a) Notebook
Stratified 80/20 split of the 150-row Iris dataset (`random_state=42`). Results on the 30-sample test set:

| Model | Accuracy | F1 (macro) |
|---|---|---|
| Random Forest (`n_estimators=10`, `max_depth=2`) — baseline | **0.9333** | **0.9333** |
| Logistic Regression (`C=1.0`) | 0.9667 | 0.9666 |

Setosa is classified perfectly; the errors are between versicolor and virginica.

### (b) Scripts and `params.yaml`
The notebook is split into three stages. Every hard-coded value now comes from `params.yaml`:

- `prepare.py` → reads `data/raw/iris.csv`, writes `data/prepared/{train,test}.csv` (`prepare.test_size`, `prepare.random_state`)
- `train.py` → fits the model and saves `models/model.joblib` (`train.*`: model type, `n_estimators`, `max_depth`, `C`, …)
- `evaluate.py` → writes `metrics.json` (accuracy, F1), `plots/predictions.csv` and `plots/confusion_matrix.png`

### (c) Git + DVC + remote
```bash
pip install -r requirements.txt
git init -b main && dvc init
dvc add data/raw/iris.csv                  # creates iris.csv.dvc, git-ignores the CSV
git add data/raw/iris.csv.dvc data/raw/.gitignore && git commit -m "Track dataset"
dvc remote add -d localremote ~/dvc-storage   # or gdrive://, s3://, DagsHub
dvc push
```

### (d) Pipeline, parameter change, comparison
```bash
dvc repro                                   # runs prepare → train → evaluate
git add -A && git commit -m "Baseline max_depth=2"
sed -i 's/max_depth: 2/max_depth: 3/' params.yaml
dvc repro                                   # only train + evaluate re-run
dvc params diff
dvc metrics diff
```
Expected output (verified by running the stages):
```
Path         Param            HEAD    workspace
params.yaml  train.max_depth  2       3

Path          Metric    HEAD     workspace    Change
metrics.json  accuracy  0.9333   0.9667       0.0334
metrics.json  f1_score  0.9333   0.9666       0.0333
```
`scripts/q1_run_experiment.sh` runs all of (c) and (d) in one go.

### (e) CML on pull requests
`.github/workflows/cml.yaml` runs on every PR to `main`: it installs dependencies, gets the data
(`dvc pull`, or regenerates the byte-identical CSV if the remote is local), runs `dvc repro`, then
builds a Markdown report with `dvc metrics diff main`, `dvc params diff main` and the confusion
matrix image, and posts it on the PR with `cml comment update`.

Try it:
```bash
git switch -c exp/max-depth-3
git commit -am "Exp: max_depth=3" && git push -u origin exp/max-depth-3
# open a PR on GitHub → the CML bot comments the report
```

---

## Question 2 — Deploying the model as a scalable service with CI/CD

### (a) API and Dockerfile
FastAPI app (`api/main.py`):
- `GET /health` → `200 {"status":"ok","model_loaded":true,"model_version":"<sha>"}`, or `503` until the model is loaded
- `POST /predict` → `{"instances":[{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}]}` returns species + class probabilities

```bash
dvc repro
docker build -t iris-api .
docker run -p 8000:8000 iris-api      # docs at http://localhost:8000/docs
```
The Dockerfile uses `python:3.11-slim`, runs as a non-root user, adds a `HEALTHCHECK`, and gets `MODEL_VERSION` (the commit SHA) as a build argument.

### (b) Kubernetes
- `k8s/deployment.yaml` — **3 replicas**, readiness + liveness probes on `/health`, CPU/memory requests and limits, rolling update with `maxUnavailable: 0`
- `k8s/service.yaml` — NodePort Service (port 80 → 8000, nodePort 30080)
- `k8s/hpa.yaml` — HPA, 3–10 pods at 50 % CPU

```bash
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl scale deployment/iris-api --replicas=5          # manual scaling
kubectl apply -f k8s/hpa.yaml                           # or: kubectl autoscale deployment iris-api --cpu-percent=50 --min=3 --max=10
```
`scripts/q2_deploy_local.sh` does all of this on minikube, including a smoke test and a load generator for the HPA.

### (c) CI: MLflow + Docker, SHA-tagged image, registry
Job `build` in `.github/workflows/cicd.yaml` (on push to `main`):
1. `dvc repro` → quality gate (fails if accuracy < 0.85) → `pytest`
2. `src/package_mlflow.py` logs params and metrics, registers the model as **`iris-classifier`** in MLflow, and saves an MLflow model package (uploaded as a workflow artifact)
3. `docker build` tagged **`ghcr.io/<owner>/<repo>:<commit-sha>`** (and `:latest`), container smoke test, then `docker push` to GitHub Container Registry using the built-in `GITHUB_TOKEN`

### (d) Continuous deployment with verification and rollback
Job `deploy` (runs only if `build` succeeds):
1. Connects to a cluster: a real one if the `KUBE_CONFIG` secret (base64 kubeconfig) is set, otherwise a temporary **kind** cluster inside the runner
2. `kubectl set image deployment/iris-api iris-api=ghcr.io/<owner>/<repo>:<sha>` and records the change-cause
3. `kubectl rollout status --timeout=120s` → on failure **`kubectl rollout undo`**
4. Smoke test through the Service checks `/health` reports the new SHA and `/predict` works → on failure **`kubectl rollout undo`**

To demonstrate rollback: **Actions → CI/CD → Run workflow → tick “simulate_failure”**. It deploys a non-existent tag, the rollout times out, and the workflow rolls back to the previous revision (the job ends red on purpose, and `kubectl rollout history` is printed).

### (e) Pipeline flow
See [`docs/pipeline.md`](docs/pipeline.md) (GitHub renders the Mermaid diagram):

Code/Data → Git + DVC → CI (CML on PRs, tests + quality gate on main) → MLflow + Docker → Registry (GHCR, SHA tag) → Kubernetes (set image → verify → rollback).

---

## One-time GitHub settings
- **Settings → Actions → General → Workflow permissions → “Read and write permissions”** (lets CML comment and the build push to GHCR).
- After the first push, the image appears under your profile's **Packages**. For a real cluster, make the package public or add an `imagePullSecret`.
