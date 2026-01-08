#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

echo "Building the Docker image..."
docker build -t test-data-app -f Dockerfile .

echo "Image built successfully."
echo "Running container to download, extract, and process datasets."
echo "Output will be mapped to /mnt/data on the host machine."

# The commands are passed to `bash -c` to be executed sequentially inside the container.
# The `&&` ensures that the next command only runs if the previous one was successful.
# All output written to /data inside the container will be saved to /mnt/data on the host.
docker run --rm -it -v /mnt/data:/data test-data-app bash -c "\
  ./bash_scripts/k400_download.sh && \
  ./bash_scripts/k400_extraction.sh && \
  python main.py && \
  echo 'All tasks complete. Data is available in /mnt/data on the host.'"