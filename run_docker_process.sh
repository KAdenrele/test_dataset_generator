#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e
docker build -t test-data-app . && docker run --rm --gpus all -v /mnt/data2/test_dataset/curated:/data