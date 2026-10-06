#!/usr/bin/env python3
"""Call the RunPod serverless endpoint and save the returned mp4.

Uses system `curl` for HTTPS (avoids broken CA stores on Homebrew Python).

Usage:
  ./scripts/run_job.py
  ./scripts/run_job.py --prompt "drone shot over a rainy neon city, vertical"
  ./scripts/run_job.py --input test_input/test_t2v.json --out out.mp4

Env (from .env or shell):
  RUNPOD_API_KEY
  RUNPOD_ENDPOINT_ID   # short id, e.g. 5mngbsk6ip48fa (full URLs OK too)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_endpoint_id(raw: str) -> str:
    """Accept bare id or common RunPod URL shapes → bare endpoint id."""
    value = raw.strip().rstrip("/")
    if not value:
        return value

    m = re.search(r"api\.runpod\.ai/v2/([^/]+)", value)
    if m:
        return m.group(1)

    m = re.search(r"https?://([a-z0-9]+)\.api\.runpod\.ai", value, re.I)
    if m:
        return m.group(1)

    if "://" in value:
        path = urlparse(value).path.strip("/")
        return path.split("/")[0] if path else value

    return value


def curl_json(
    url: str,
    *,
    api_key: str,
    timeout: int,
    payload: Optional[dict] = None,
) -> dict:
    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST" if payload is not None else "GET",
        url,
        "-H",
        f"Authorization: Bearer {api_key}",
        "--max-time",
        str(timeout),
    ]
    body_path = None
    if payload is not None:
        cmd += ["-H", "Content-Type: application/json"]
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        body_path = Path(tmp.name)
        with tmp:
            json.dump(payload, tmp)
        cmd += ["--data-binary", f"@{body_path}"]

    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    finally:
        if body_path is not None:
            body_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise SystemExit(f"curl failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout}")

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Non-JSON response:\n{proc.stdout[:2000]}") from exc


def extract_output(resp: dict) -> dict:
    if "output" in resp and isinstance(resp["output"], dict):
        return resp["output"]
    return resp


def main() -> None:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run one video job on RunPod")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "test_input" / "test_t2v.json",
        help='JSON file with {"input": {...}}',
    )
    parser.add_argument("--prompt", help="Override prompt")
    parser.add_argument("--image-url", help="Optional image URL for I2V")
    parser.add_argument("--out", type=Path, default=ROOT / "out.mp4")
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use /run + poll instead of /runsync",
    )
    parser.add_argument("--timeout", type=int, default=1800, help="HTTP timeout seconds")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    endpoint_raw = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
    if not api_key or not endpoint_raw:
        raise SystemExit(
            "Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in .env or your shell.\n"
            "Copy .env.example → .env and fill them in."
        )

    endpoint_id = normalize_endpoint_id(endpoint_raw)
    if endpoint_id != endpoint_raw:
        print(f"[run] normalized endpoint id → {endpoint_id}")

    payload = json.loads(args.input.read_text())
    if "input" not in payload:
        payload = {"input": payload}
    if args.prompt:
        payload["input"]["prompt"] = args.prompt
    if args.image_url:
        payload["input"]["image_url"] = args.image_url

    base = f"https://api.runpod.ai/v2/{endpoint_id}"
    print(f"[run] url={base} async={args.use_async}")
    print(f"[run] prompt={payload['input'].get('prompt', '')[:80]!r}")

    t0 = time.time()
    if args.use_async:
        started = curl_json(f"{base}/run", api_key=api_key, timeout=60, payload=payload)
        job_id = started.get("id")
        if not job_id:
            raise SystemExit(f"No job id in response: {started}")
        print(f"[run] job_id={job_id} — polling…")
        while True:
            status = curl_json(f"{base}/status/{job_id}", api_key=api_key, timeout=60)
            state = status.get("status")
            print(f"  status={state} ({time.time() - t0:.0f}s)")
            if state in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
                resp = status
                break
            time.sleep(5)
    else:
        resp = curl_json(
            f"{base}/runsync",
            api_key=api_key,
            timeout=args.timeout,
            payload=payload,
        )

    out = extract_output(resp)
    if out.get("error") or resp.get("status") in {"FAILED", "CANCELLED", "TIMED_OUT"}:
        raise SystemExit(f"Job failed: {json.dumps(resp, indent=2)[:2000]}")

    b64 = out.get("video_base64")
    if not b64:
        raise SystemExit(f"No video_base64 in response:\n{json.dumps(resp, indent=2)[:2000]}")

    args.out.write_bytes(base64.b64decode(b64))
    print(f"[done] wrote {args.out} ({args.out.stat().st_size} bytes) in {time.time() - t0:.1f}s")
    print(
        f"       seed={out.get('seed')} backend={out.get('backend')} "
        f"elapsed={out.get('elapsed_seconds')}s"
    )


if __name__ == "__main__":
    main()
