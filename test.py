import os
from datasets import load_dataset
from scripts.image_transform_pipeline import run_pipeline, ALL_SIMULATIONS
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = "./data"
DEST_BASE = os.path.join(BASE_DIR, "curated")

run_pipeline(
    dataset_name="SAFE",
    image_directory_path=os.path.join(BASE_DIR, "raw/safe_images_synthetic"),
    destination_directory=os.path.join(DEST_BASE, "SAFE"),
    is_huggingface=False,
    has_subdirectories=True,
    is_synthetic=True,
    simulations_to_run=ALL_SIMULATIONS,
    target_sample_size=2
)

run_pipeline(
    dataset_name="COCO",
    image_directory_path=os.path.join(BASE_DIR, "raw/coco_images_authentic"),
    destination_directory=os.path.join(DEST_BASE, "COCO"),
    is_huggingface=True,
    has_subdirectories=False,
    is_synthetic=False,
    simulations_to_run=ALL_SIMULATIONS,
    hf_name="detection-datasets/coco",
    target_sample_size=2
)
logging.info("--- Main process complete. All datasets downloaded and curated. ---")
