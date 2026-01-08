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
docker run --rm -it -v /mnt/data:/data test-data-app bash -c '
  set -e
  K400_EXTRACT_DIR="/data/raw/k400/val"
  
  echo "Checking for existing Kinetics-400 data..."
  if [ ! -d "$K400_EXTRACT_DIR" ] || [ -z "$(ls -A "$K400_EXTRACT_DIR")" ]; then
    echo "Kinetics-400 data not found or is empty. Starting download and extraction..."
    ./bash_scripts/k400_download.sh && ./bash_scripts/k400_extraction.sh
  else
    echo "Kinetics-400 data already exists. Skipping download and extraction."
  fi
  
  echo "Running main Python script (will use cached COCO data if available)..."
  python main.py
  
  echo "All tasks complete. Data is available in /mnt/data on the host."'