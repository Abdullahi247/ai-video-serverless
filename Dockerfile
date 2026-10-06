# AI Video Serverless — RunPod queue worker
# Build for linux/amd64 (RunPod GPU workers are x86_64 + NVIDIA).
#
#   docker build --platform linux/amd64 -t YOUR_DOCKERHUB_USER/ai-video-serverless:v1 .
#   docker push YOUR_DOCKERHUB_USER/ai-video-serverless:v1
#
# Default model path: Wan 2.2 TI2V-5B (fits ~24GB with offload).
# For MiniMax H3 / LTX-2.5, use a 48GB+ GPU and set MODEL_BACKEND accordingly.

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MODEL_BACKEND=wan22 \
    MODEL_ID=Wan-AI/Wan2.2-TI2V-5B \
    MODEL_CACHE_DIR=/runpod-volume/models \
    HF_HOME=/runpod-volume/huggingface \
    OUTPUT_DIR=/tmp/outputs \
    TORCH_DTYPE=float16

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        ffmpeg \
        git \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

COPY requirements.txt /app/requirements.txt

# PyTorch >= 2.6 required (diffusers/accelerate use torch.accelerator).
# Install CUDA wheels first and keep them out of requirements.txt so a
# later pip install cannot overwrite them with a CPU build.
RUN pip3 install --upgrade pip \
    && pip3 install \
        torch==2.6.0 \
        torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install -r /app/requirements.txt \
    && python -c "import torch; assert hasattr(torch, 'accelerator'), torch.__version__; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

COPY src/ /app/src/
COPY scripts/ /app/scripts/

RUN mkdir -p /tmp/outputs /runpod-volume/models /runpod-volume/huggingface \
    && chmod +x /app/scripts/*.sh

# Models are downloaded at cold start (or from network volume) — keep image lean.
CMD ["python", "-u", "/app/src/handler.py"]
