import os
from datasets import load_dataset
from scripts.image_transform_pipeline import run_pipeline

run_pipeline("COCO", target_sample_size=2000)
