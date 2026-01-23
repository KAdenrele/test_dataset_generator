FROM ghcr.io/astral-sh/uv:latest AS uv_bin

FROM python:3.11-slim
# Use a GPU-enabled base image from NVIDIA to support the `--gpus all` flag.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
# Set environment variables to prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies.
# - The deadsnakes PPA is used to get Python 3.11 on Ubuntu.
# - `ffmpeg` is required for video processing in `media_processes.py`.
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv_bin /uv /uvx /bin/
COPY --from=uv_bin /uv /usr/local/bin/

COPY pyproject.toml uv.lock ./

# --frozen: ensures uv doesn't try to update the lockfile
# --no-cache: keeps the image size small
# Create a virtual environment with Python 3.11 and install dependencies.
RUN python3.11 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"
RUN uv sync --frozen --no-cache
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

VOLUME /data

CMD ["python", "main.py"]