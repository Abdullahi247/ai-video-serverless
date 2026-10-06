"""
RunPod Serverless handler for text/image → video generation.

Cold start loads the model once; each job is a short clip request.

Input schema (job["input"]):
  {
    "prompt": "a footballer scoring a goal, cinematic",
    "negative_prompt": "blurry, low quality",          # optional
    "image_url": "https://...",                       # optional — enables I2V
    "duration_seconds": 5,                            # 1–10 typical
    "width": 720,
    "height": 1280,                                   # 9:16 default for Shorts
    "seed": 42,                                       # optional
    "num_inference_steps": 30,                        # optional
    "guidance_scale": 5.0                             # optional
  }

Output:
  {
    "video_base64": "...",          # mp4 (small clips)
    "seed": 42,
    "backend": "wan22",
    "duration_seconds": 5,
    "resolution": "720x1280"
  }

Env:
  MODEL_BACKEND = wan22 | stub
  MODEL_ID      = Hugging Face repo id
  MODEL_CACHE_DIR, HF_HOME, OUTPUT_DIR, TORCH_DTYPE
"""

from __future__ import annotations

import base64
import os
import time
import traceback
from pathlib import Path
from typing import Any

import runpod

from backends import get_backend

BACKEND_NAME = os.getenv("MODEL_BACKEND", "wan22").strip().lower()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[boot] MODEL_BACKEND={BACKEND_NAME}")
_t0 = time.time()
try:
    BACKEND = get_backend(BACKEND_NAME)
    print(f"[boot] backend ready in {time.time() - _t0:.1f}s")
except Exception as exc:  # noqa: BLE001 — surface cold-start errors clearly
    print(f"[boot] FAILED to load backend: {exc}")
    traceback.print_exc()
    BACKEND = None
    BOOT_ERROR = str(exc)
else:
    BOOT_ERROR = None


def _clamp_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _clamp_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def handler(job: dict) -> dict:
    if BOOT_ERROR or BACKEND is None:
        return {"error": f"Worker failed to load model: {BOOT_ERROR}"}

    job_input = job.get("input") or {}
    prompt = (job_input.get("prompt") or "").strip()
    if not prompt and not job_input.get("image_url"):
        return {"error": "Provide at least 'prompt' or 'image_url'."}

    width = _clamp_int(job_input.get("width"), 720, 256, 1280)
    height = _clamp_int(job_input.get("height"), 1280, 256, 1280)
    # Keep multiples of 16 for latent models
    width -= width % 16
    height -= height % 16

    duration = _clamp_float(job_input.get("duration_seconds"), 5.0, 1.0, 10.0)
    steps = _clamp_int(job_input.get("num_inference_steps"), 30, 4, 60)
    guidance = _clamp_float(job_input.get("guidance_scale"), 5.0, 1.0, 15.0)
    seed = job_input.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None

    job_id = job.get("id", "local")
    out_path = OUTPUT_DIR / f"{job_id}.mp4"

    t0 = time.time()
    try:
        result = BACKEND.generate(
            prompt=prompt or "cinematic motion",
            negative_prompt=job_input.get("negative_prompt") or "",
            image_url=job_input.get("image_url"),
            width=width,
            height=height,
            duration_seconds=duration,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=seed,
            output_path=out_path,
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return {"error": f"Generation failed: {exc}"}

    if not out_path.exists():
        return {"error": "Model finished but no output file was written."}

    video_b64 = base64.b64encode(out_path.read_bytes()).decode("ascii")
    # Keep disk clean on ephemeral workers
    try:
        out_path.unlink(missing_ok=True)
    except OSError:
        pass

    return {
        "video_base64": video_b64,
        "seed": result.get("seed"),
        "backend": BACKEND_NAME,
        "duration_seconds": duration,
        "resolution": f"{width}x{height}",
        "elapsed_seconds": round(time.time() - t0, 2),
    }


runpod.serverless.start({"handler": handler})
