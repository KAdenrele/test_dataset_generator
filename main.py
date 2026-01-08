import os
from datasets import load_dataset
# coco_dir = os.path.join(os.path.curdir, "authentic/images/COCO")
# ulta_video__dir = os.path.join(os.path.curdir, "authentic/videos/ultra_video")

#ultra_video_dataset = load_dataset("APRIL-AIGC/UltraVideo")
#clip_kinetics_700_dataset = load_dataset("iejMac/CLIP-Kinetics700")
coco_dataset = load_dataset("detection-datasets/coco")

def preprocess_function(examples):
    #no preprocessing for now, just return the media. 
    return examples

coco_processed_dataset = coco_dataset.map(preprocess_function,batched=True, num_proc=os.cpu_count(),desc="Resizing images", load_from_cache_file=True)
#ultra_video_processed_dataset = ultra_video_dataset.map(preprocess_function,batched=True, num_proc=os.cpu_count(),desc="Resizing images", load_from_cache_file=True)
#clip_kinetics_700_processed_dataset = clip_kinetics_700_dataset.map(preprocess_function,batched=True, num_proc=os.cpu_count(),desc="Resizing images", load_from_cache_file=True)