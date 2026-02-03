#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e
# Temporarily disable BuildKit to work around a broken buildx component in the Docker environment.
# This forces Docker to use the legacy builder.
#DOCKER_BUILDKIT=0 docker build -t test-data-app . && docker run --rm --name data-gen-container -v /mnt/:/data test-data-app

docker build -t test-data-app . && docker run --rm --name data-gen-container -v /mnt/data/test_dataset:/data test-data-app