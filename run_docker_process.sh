#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e
docker build -t test-data-app . && docker run --rm --name data-gen-container -v /mnt/:/data test-data-app