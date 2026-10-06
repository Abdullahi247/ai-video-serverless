#!/usr/bin/env bash
# Build + push linux/amd64 image for RunPod Serverless.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

USER_NAME="${DOCKERHUB_USER:-YOUR_DOCKERHUB_USER}"
TAG="${IMAGE_TAG:-v1}"
IMAGE="${USER_NAME}/ai-video-serverless:${TAG}"

echo "Building ${IMAGE} (linux/amd64)…"
docker build --platform linux/amd64 -t "$IMAGE" .

echo "Pushing ${IMAGE}…"
docker push "$IMAGE"

cat <<EOF

Done. In RunPod Console:
  1. Serverless → New Endpoint → Custom GPU Worker (or Template from image)
  2. Container image: ${IMAGE}
  3. GPU: RTX 4090 (24GB) minimum for Wan 2.2 TI2V-5B; 48GB+ preferred
  4. Env:
       MODEL_BACKEND=wan22
       MODEL_ID=Wan-AI/Wan2.2-TI2V-5B
  5. Attach a Network Volume at /runpod-volume to cache HF weights across cold starts
  6. Workers min=0, max=1..N (scale to zero)

Test:
  curl -X POST "https://api.runpod.ai/v2/\$ENDPOINT_ID/runsync" \\
    -H "Authorization: Bearer \$RUNPOD_API_KEY" \\
    -H "Content-Type: application/json" \\
    -d @test_input/test_t2v.json
EOF
