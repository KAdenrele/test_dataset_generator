FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends wget && \
    pip install uv && \
    rm -rf /var/lib/apt/lists/*


COPY uv.lock .

RUN uv pip sync --lockfile uv.lock

COPY . .

VOLUME /data

# Specify the command to run on container startup
CMD ["python", "main.py"]