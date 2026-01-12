import os
from datasets import load_dataset
from scripts.image_transform_pipeline import run_pipeline

run_pipeline("SAFE", target_sample_size=1)
run_pipeline("COCO", target_sample_size=1)
print("--- Main process complete. All datasets downloaded and curated. ---")
