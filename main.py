import os
from datasets import load_dataset
from scripts.image_transform_pipeline import run_pipeline, ALL_SIMULATIONS
from scripts.video_transform_pipeline import run_pipeline as run_video_pipeline, ALL_SIMULATIONS as VIDEO_SIMULATIONS
import logging

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = "/data"
DEST_BASE = os.path.join(BASE_DIR, "data/test_dataset/curated/images")
VIDEO_DEST_BASE = os.path.join(BASE_DIR, "data/test_dataset/curated/videos")

if __name__ == "__main__":
  # run_pipeline(
  #     dataset_name="SAFE",
  #     image_directory_path=os.path.join(BASE_DIR, "data2/training_data/SAFE/data"),
  #     destination_directory=os.path.join(DEST_BASE, "SAFE"),
  #     is_huggingface=False,
  #     has_subdirectories=True,
  #     is_synthetic=True,
  #     simulations_to_run=["original"],
  #     target_sample_size=2000,
  #     max_workers=3
  # )
    
  # run_pipeline(
  #   dataset_name="COCO",
  #   image_directory_path=os.path.join(BASE_DIR, "data/test_dataset/raw/coco_images_authentic"),
  #   destination_directory=os.path.join(DEST_BASE, "COCO"),
  #   is_huggingface=True,
  #   has_subdirectories=False,
  #   is_synthetic=False,
  #   simulations_to_run=["original"],
  #   hf_name="detection-datasets/coco",
  #   target_sample_size=2000,
  #   max_workers=3
  # )
     
  # run_pipeline(
  #     dataset_name="Inswapper",
  #     image_directory_path=os.path.join(BASE_DIR, "data/test_dataset/raw/inswapper_images"),
  #     destination_directory=os.path.join(DEST_BASE, "Inswapper"),
  #     is_huggingface=False,
  #     has_subdirectories=False,
  #     is_synthetic=True,
  #     simulations_to_run=["original"],
  #     target_sample_size=2000,
  #     max_workers=3
  # )


 run_video_pipeline(
        dataset_name="MAVOS",
        video_directory_path=os.path.join(BASE_DIR, "raw/MAVOS_DD"),
        destination_directory=os.path.join(VIDEO_DEST_BASE, "MAVOS_DD"),
        is_huggingface=True,
        has_subdirectories=True,
        hf_name="unibuc-cs/MAVOS-DD",
        is_synthetic=True,
        simulations_to_run=VIDEO_SIMULATIONS[3:],
        target_sample_size=2000,
    )
 run_video_pipeline(
        dataset_name="K400",
        video_directory_path=os.path.join(BASE_DIR, "raw/k400/val"),
        destination_directory=os.path.join(VIDEO_DEST_BASE, "K400"),
        is_huggingface=False,
        has_subdirectories=False,
        is_synthetic=False,
        simulations_to_run=VIDEO_SIMULATIONS[3:],
        target_sample_size=2000,
    )