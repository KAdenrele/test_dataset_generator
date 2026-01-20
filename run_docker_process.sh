#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

docker build -t test-data-app . && \
docker run -d \
  --name test-data-proc \
  --restart unless-stopped \
  -v /mnt:/data \
  test-data-app
