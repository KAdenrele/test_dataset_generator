import os
from datasets import load_dataset
from scripts.image_transform_pipeline import run_pipeline

BASE_DOWNLOAD_DIR = "/data/raw"
COCO_DIR = f"{BASE_DOWNLOAD_DIR}/coco_images_authentic"
MAVOS_DIR = f"{BASE_DOWNLOAD_DIR}/mavos_videos_synthetic"


if __name__ == "__main__":
    coco_dataset = load_dataset("detection-datasets/coco", cache_dir=COCO_DIR, num_proc=os.cpu_count())
    #mavos_dataset = load_dataset("unibuc-cs/MAVOS-DD", cache_dir=MAVOS_DIR, num_proc=os.cpu_count())

    #run_pipeline("SAFE", target_sample_size=2000)
    run_pipeline("COCO", target_sample_size=2000)

    print("--- Main process complete. All datasets downloaded and curated. ---")