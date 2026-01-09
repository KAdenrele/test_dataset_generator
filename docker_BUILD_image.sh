#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

echo "Building the Docker image..."
docker build -t test-data-app -f Dockerfile .
#docker buildx build --tag test-data-app --file Dockerfile --load .

echo "Image built successfully."