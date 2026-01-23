FROM ghcr.io/astral-sh/uv:latest AS uv_bin

# Use a GPU-enabled base image from NVIDIA to support potential GPU operations and install ffmpeg.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

# Set environment variables to prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies.
# - The deadsnakes PPA is used to get Python 3.11 on Ubuntu 22.04.
# - `ffmpeg` is required for video processing in `media_processes.py`.
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv_bin /uv /usr/local/bin/

COPY pyproject.toml uv.lock ./

# --frozen: ensures uv doesn't try to update the lockfile
# --no-cache: keeps the image size small
# Create a virtual environment and install dependencies
RUN python3.11 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"
RUN uv sync --frozen --no-cache

COPY . .

VOLUME /data

CMD ["python", "main.py"]