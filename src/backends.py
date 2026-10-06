"""Video generation backends for the RunPod worker.

- stub  : no GPU model — writes a short placeholder mp4 (local/CI smoke tests)
- wan22 : Wan 2.2 TI2V-5B via Diffusers (practical on ~24GB with offload)

MiniMax H3 / LTX-2.5 can be added as extra backends when you have 48GB+ GPUs
and have accepted their licenses on Hugging Face.
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class VideoBackend(ABC):
    @abstractmethod
    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        image_url: Optional[str],
        width: int,
        height: int,
        duration_seconds: float,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int],
        output_path: Path,
    ) -> dict[str, Any]:
        raise NotImplementedError


class StubBackend(VideoBackend):
    """FFmpeg color bars + drawtext — proves the serverless path without a GPU model."""

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        image_url: Optional[str],
        width: int,
        height: int,
        duration_seconds: float,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int],
        output_path: Path,
    ) -> dict[str, Any]:
        # Avoid drawtext — many ffmpeg builds (incl. stock brew) lack libfreetype.
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={width}x{height}:rate=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-t",
            str(duration_seconds),
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        _ = prompt  # reserved for future overlay / logging
        return {"seed": seed if seed is not None else 0}


class Wan22Backend(VideoBackend):
    """Wan 2.2 text/image → video (Diffusers).

    Requires enough VRAM for the selected MODEL_ID. TI2V-5B is the default
    for 24GB cards; A14B MoE variants want 48–80GB-class GPUs.
    """

    def __init__(self) -> None:
        import torch
        from diffusers.utils import export_to_video

        self.torch = torch
        self.export_to_video = export_to_video
        self.model_id = os.getenv("MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B")
        self.cache_dir = os.getenv("MODEL_CACHE_DIR", "/runpod-volume/models")
        dtype_name = os.getenv("TORCH_DTYPE", "float16").lower()
        self.dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype_name, torch.float16)

        print(f"[wan22] loading {self.model_id} dtype={self.dtype} …")
        # Import late so stub workers don't need the full stack at parse time
        try:
            from diffusers import WanPipeline  # type: ignore
        except ImportError:
            # Older / newer diffusers may expose a different class name
            from diffusers import DiffusionPipeline as WanPipeline  # type: ignore

        pipe_kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "cache_dir": self.cache_dir,
        }
        try:
            self.pipe = WanPipeline.from_pretrained(self.model_id, **pipe_kwargs)
        except Exception:
            # Fallback: generic DiffusionPipeline if WanPipeline isn't wired yet
            from diffusers import DiffusionPipeline

            self.pipe = DiffusionPipeline.from_pretrained(self.model_id, **pipe_kwargs)

        if torch.cuda.is_available():
            # Prefer offload on 24GB cards; full GPU on 48GB+
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 40:
                print(f"[wan22] {vram_gb:.1f}GB VRAM → enable_model_cpu_offload()")
                self.pipe.enable_model_cpu_offload()
            else:
                print(f"[wan22] {vram_gb:.1f}GB VRAM → .to('cuda')")
                self.pipe.to("cuda")
            try:
                self.pipe.vae.enable_tiling()
            except Exception:
                pass
        else:
            print("[wan22] WARNING: no CUDA — will be extremely slow on CPU")
            self.pipe.to("cpu")

        self._fps = int(os.getenv("VIDEO_FPS", "16"))

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        image_url: Optional[str],
        width: int,
        height: int,
        duration_seconds: float,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int],
        output_path: Path,
    ) -> dict[str, Any]:
        torch = self.torch
        generator = None
        used_seed = seed if seed is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(used_seed)

        num_frames = max(8, int(duration_seconds * self._fps))
        # Many Wan pipelines expect frame counts aligned to 4k+1
        num_frames = ((num_frames - 1) // 4) * 4 + 1

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }

        if image_url:
            from diffusers.utils import load_image

            call_kwargs["image"] = load_image(image_url)

        # Drop None kwargs some pipelines reject
        call_kwargs = {k: v for k, v in call_kwargs.items() if v is not None}

        print(f"[wan22] generate frames={num_frames} size={width}x{height} seed={used_seed}")
        result = self.pipe(**call_kwargs)
        frames = result.frames[0] if hasattr(result, "frames") else result.images
        self.export_to_video(frames, str(output_path), fps=self._fps)
        return {"seed": used_seed}


def get_backend(name: str) -> VideoBackend:
    key = (name or "wan22").strip().lower()
    if key in {"stub", "dummy", "test"}:
        return StubBackend()
    if key in {"wan22", "wan", "wan2.2"}:
        return Wan22Backend()
    if key in {"minimax", "minimax-h3", "h3"}:
        raise NotImplementedError(
            "MiniMax H3 backend is scaffolded for a follow-up. "
            "Use MODEL_BACKEND=wan22 on 24GB, or implement H3 once you have "
            "48GB+ GPUs and accepted the MiniMax community license."
        )
    raise ValueError(f"Unknown MODEL_BACKEND={name!r}. Use: stub | wan22")
