#!/usr/bin/env bash
# Q1 (c) + (d): initialise Git & DVC, track the data, configure a remote,
# run the pipeline, change a parameter and compare the two runs.
#
# Run once from the repo root (fresh folder, before the first push):
#   pip install -r requirements.txt
#   bash scripts/q1_run_experiment.sh
set -euo pipefail

REMOTE_DIR="${DVC_REMOTE_DIR:-$HOME/dvc-storage}"   # local remote for the demo

echo "=== (c) 1. Initialise Git and DVC ==="
[ -d .git ] || git init -b main
[ -d .dvc ] || dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC" || true

echo "=== (c) 2. Track the dataset with DVC ==="
[ -f data/raw/iris.csv ] || python src/get_data.py
dvc add data/raw/iris.csv          # -> data/raw/iris.csv.dvc (+ data/raw/.gitignore)
git add data/raw/iris.csv.dvc data/raw/.gitignore
git commit -m "Track raw Iris dataset with DVC"

echo "=== (c) 3. Configure a DVC remote and push the data ==="
mkdir -p "$REMOTE_DIR"
dvc remote add -d -f localremote "$REMOTE_DIR"
# Cloud alternatives (then `dvc push`):
#   dvc remote add -d -f gdrive   gdrive://<folder-id>
#   dvc remote add -d -f s3remote s3://<bucket>/dvc
#   dvc remote add -d -f dagshub  https://dagshub.com/<user>/<repo>.dvc
git add .dvc/config
git commit -m "Configure DVC remote"
dvc push

echo "=== (d) 4. Baseline run (max_depth=2) ==="
dvc repro
dvc metrics show
git add -A                         # code, dvc.yaml, dvc.lock, params, metrics, plots, workflows
git commit -m "Baseline experiment: random_forest max_depth=2"
dvc push

echo "=== (d) 5. Change a parameter: train.max_depth 2 -> 3 ==="
sed -i.bak 's/max_depth: 2/max_depth: 3/' params.yaml && rm -f params.yaml.bak
dvc repro

echo "=== (d) 6. Compare workspace vs last commit ==="
echo "--- dvc params diff ---"
dvc params diff
echo "--- dvc metrics diff ---"
dvc metrics diff

echo
echo "Experiment left uncommitted on purpose. Keep it on a branch for the CML PR:"
echo "  git switch -c exp/max-depth-3 && git commit -am 'Exp: max_depth=3' && dvc push"
