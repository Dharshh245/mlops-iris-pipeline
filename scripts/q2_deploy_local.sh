#!/usr/bin/env bash
# Q2 (b) - Build the image, deploy to a local cluster (minikube), and scale it.
# Requires: docker, minikube, kubectl, and a trained model (`dvc repro`).
set -euo pipefail

IMAGE="ghcr.io/dharshh245/mlops-iris-pipeline:latest"

[ -f models/model.joblib ] || dvc repro

echo "=== Build image inside minikube's Docker daemon ==="
eval "$(minikube docker-env)"
docker build -t "$IMAGE" --build-arg MODEL_VERSION=local .

echo "=== Deploy: Deployment (3 replicas + probes) and Service ==="
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl rollout status deployment/iris-api --timeout=120s
kubectl get pods -l app=iris-api -o wide

echo "=== Smoke test ==="
URL="$(minikube service iris-api --url)"
curl -s "$URL/health"; echo
curl -s -X POST "$URL/predict" -H 'Content-Type: application/json' \
  -d '{"instances":[{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2},
                    {"sepal_length":6.7,"sepal_width":3.0,"petal_length":5.2,"petal_width":2.3}]}'
echo

echo "=== Scale manually: 3 -> 5 replicas ==="
kubectl scale deployment/iris-api --replicas=5
kubectl rollout status deployment/iris-api
kubectl get deployment iris-api

echo "=== Autoscale with an HPA (3..10 pods at 50% CPU) ==="
minikube addons enable metrics-server
kubectl apply -f k8s/hpa.yaml
# Equivalent one-liner: kubectl autoscale deployment iris-api --cpu-percent=50 --min=3 --max=10
kubectl get hpa iris-api

cat <<'EOF'

Generate load to watch the HPA scale out (in a second terminal):
  kubectl run load --rm -it --image=busybox --restart=Never -- /bin/sh -c \
    'while true; do wget -q -O- --post-data="{\"instances\":[{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}]}" --header="Content-Type: application/json" http://iris-api/predict >/dev/null; done'
  kubectl get hpa iris-api --watch
EOF
