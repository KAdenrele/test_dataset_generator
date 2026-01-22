import os
from datasets import load_dataset
from scripts.image_transform_pipeline import run_pipeline, ALL_SIMULATIONS
import logging

#MAVOS_DIR = f"{BASE_DOWNLOAD_DIR}/mavos_videos_synthetic"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = "/data"
DEST_BASE = os.path.join(BASE_DIR, "test_dataset/curated/images")

if __name__ == "__main__":
    run_pipeline(
        dataset_name="SAFE",
        image_directory_path=os.path.join(BASE_DIR, "training_data/SAFE/data"),
        destination_directory=os.path.join(DEST_BASE, "SAFE"),
        is_huggingface=False,
        has_subdirectories=True,
        is_synthetic=True,
        simulations_to_run=ALL_SIMULATIONS,
        target_sample_size=2000
    )
    run_pipeline(
        dataset_name="COCO",
        image_directory_path=os.path.join(BASE_DIR, "test_dataset/raw/coco_images_authentic"),
        destination_directory=os.path.join(DEST_BASE, "COCO"),
        is_huggingface=True,
        has_subdirectories=False,
        is_synthetic=False,
        simulations_to_run=ALL_SIMULATIONS,
        hf_name="detection-datasets/coco",
        target_sample_size=2000
    )
    run_pipeline(
        dataset_name="Inswapper",
        image_directory_path=os.path.join(BASE_DIR, "test_dataset/raw/inswapper_images"),
        destination_directory=os.path.join(DEST_BASE, "Inswapper"),
        is_huggingface=False,
        has_subdirectories=False,
        is_synthetic=True,
        simulations_to_run=ALL_SIMULATIONS,
        target_sample_size=2000
    )

    logging.info("--- Main process complete. All datasets downloaded and curated. ---")