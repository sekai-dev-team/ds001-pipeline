#!/usr/bin/env bash
# Build and push Docker image for DS-001 Pipeline v0.2
# Usage: ./build-and-push.sh
set -euo pipefail

IMAGE="kona01zz/ds001-pipeline:latest"

echo "==> Building Docker image: ${IMAGE}"
docker build -t "${IMAGE}" .

echo "==> Pushing to registry: ${IMAGE}"
docker push "${IMAGE}"

echo "==> Done: ${IMAGE}"
