# AI Video Serverless (RunPod)

Queue-based **RunPod Serverless** worker that generates short vertical videos from a text prompt (optional image). Built for GPU cold-start → generate → scale-to-zero.

## Why this layout

| Piece | Role |
|-------|------|
| `Dockerfile` | CUDA 12.1 + PyTorch + Diffusers worker image (`linux/amd64`) |
| `Dockerfile.stub` | CPU-only smoke image (no NVIDIA on your laptop) |
| `src/handler.py` | RunPod job contract (`job["input"]` → mp4 base64) |
| `src/backends.py` | `stub` (ffmpeg) + `wan22` (Wan 2.2 TI2V-5B) |
| `scripts/build_and_push.sh` | Build/push to Docker Hub for RunPod |

**Default model = Wan 2.2 TI2V-5B** — Apache 2.0, realistic on a **24GB** card with CPU offload.  
**MiniMax H3** is the quality leader but needs **48GB+** and a restricted community license; the backend hook is left for a follow-up (`MODEL_BACKEND=minimax` raises a clear error today).

## Quick start (local stub — no GPU)

```bash
# If still under video_analyzer_raw, move once:
#   mv ~/Desktop/video_analyzer_raw/ai_video_serverless ~/Desktop/

cd ~/Desktop/ai_video_serverless   # or: cd /path/to/ai_video_serverless
chmod +x scripts/*.sh
./scripts/local_stub_test.sh
# or:
docker compose run --rm worker-stub
```

## Build for RunPod

**Preferred: GitHub Actions** (native `linux/amd64` — much faster than Colima on a Mac).

1. Push this repo to GitHub.
2. Add secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` (Hub → Account Settings → Security → Access Token).
3. Push to `main`, or **Actions → Docker publish → Run workflow**.
4. Image: `YOUR_USER/ai-video-serverless:latest` (also tagged with the commit sha).

Local fallback (slow on Apple Silicon):

```bash
export DOCKERHUB_USER=your_user
./scripts/build_and_push.sh
```

Then in [RunPod Console](https://www.runpod.io) → **Serverless** → new endpoint from your image:

| Setting | Suggested |
|---------|-----------|
| GPU | RTX 4090 (24GB) min · L40S / A100 better |
| Container disk | ≥ 40 GB |
| Network volume | Mount at `/runpod-volume` (caches HF weights) |
| Workers min / max | `0` / `1+` (scale to zero) |
| Env | `MODEL_BACKEND=wan22` · `MODEL_ID=Wan-AI/Wan2.2-TI2V-5B` |

### Approximate RunPod GPU rates (bill per second of GPU time)

| GPU | Secure Pod $/hr | ≈ $/sec | Serverless $/hr | ≈ $/sec |
|-----|-----------------|---------|-----------------|---------|
| RTX 4090 24GB | $0.74 | $0.000206 | $1.10 | $0.000306 |
| L40S 48GB | $1.09 | $0.000303 | $1.75 | $0.000486 |
| A100 80GB | $1.59 | $0.000442 | $2.72 | $0.000756 |

A 5s clip can take **minutes** of GPU time — cost = GPU-seconds × rate, not “5 × rate”.

## API shape

**Request** (`POST /run` or `/runsync`):

```json
{
  "input": {
    "prompt": "cinematic football goal, stadium lights, vertical",
    "negative_prompt": "blurry, watermark",
    "duration_seconds": 5,
    "width": 720,
    "height": 1280,
    "seed": 42
  }
}
```

**Response** (success):

```json
{
  "video_base64": "<mp4 bytes>",
  "seed": 42,
  "backend": "wan22",
  "duration_seconds": 5,
  "resolution": "720x1280",
  "elapsed_seconds": 187.4
}
```

Decode locally:

```bash
python -c "import base64,sys; open('out.mp4','wb').write(base64.b64decode(sys.argv[1]))" "$B64"
```

## Architecture

```
Client → RunPod Serverless API → GPU worker (this image)
                                      │
                                      ├─ cold start: load Wan / stub once
                                      ├─ job: prompt → frames → ffmpeg/export mp4
                                      └─ return base64 (or swap to S3 URL later)
```

## Next steps

1. Smoke-test with `MODEL_BACKEND=stub`
2. Push CUDA image, attach network volume, run one Wan job on a 4090
3. Add S3/R2 upload instead of base64 for longer clips
4. Optional: implement `MiniMaxBackend` once you have H3 license + 48GB+ GPUs

## License note

- This repo scaffold: use freely
- **Wan 2.2** weights: Apache 2.0
- **MiniMax H3** / **LTX-2.5**: community licenses with geo/revenue limits — read before commercial use
