#!/usr/bin/env bash
# Local smoke test without building the full CUDA image.
# Uses MODEL_BACKEND=stub so no GPU / HF weights are required.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q runpod imageio imageio-ffmpeg opencv-python-headless Pillow numpy requests

export MODEL_BACKEND=stub
export OUTPUT_DIR="$ROOT/test_input/out"
mkdir -p "$OUTPUT_DIR"

python -u src/handler.py --test_input "$(cat "$ROOT/test_input/test_t2v.json")"
echo "Stub local test finished."
