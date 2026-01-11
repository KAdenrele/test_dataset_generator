import os
import shutil
from glob import glob
from tqdm import tqdm
import csv
import random
from datasets import load_dataset
from scripts.media_processes import SocialMediaSimulator

random.seed(42)  
BASE_DIR = "/data"

DATASET_DIRS = {
    "SAFE": os.path.join(BASE_DIR, "SAFEDataset/data"),
    "COCO": os.path.join(BASE_DIR, "raw/coco_images_authentic")
}

CURATED_DIR = os.path.join(BASE_DIR, "curated/images")
os.makedirs(CURATED_DIR, exist_ok=True)

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.tiff', '.heic']


image_storage_type_dict = {
    "SAFE": {"type": "synthetic", "storage": "images", "hf_name": None},
    "COCO": {"type": "authentic", "storage": "hugging_face_lock_file", "hf_name": "detection-datasets/coco"}
}

def get_media_info(file_path, dataset_name, base_dir):
    """Extracts metadata from the file path and dataset info."""
    media_type = "image"
    authenticity = image_storage_type_dict.get(dataset_name, {}).get("type", "unknown")
    original_filename = os.path.basename(file_path)

    source_model = None
    if dataset_name == "SAFE":
        try:
            relative_dir_path = os.path.relpath(os.path.dirname(file_path), base_dir)
            # The model name is the first component of this relative path... he says... hopefully
            source_model = relative_dir_path.split(os.sep)[0]
        except ValueError:
            source_model = "unknown" # Should not happen if paths are correct
    
    return media_type, authenticity, original_filename, source_model

def get_hf_dataset_paths(hf_name, cache_dir, target_sample_size, split='val', image_class="person"):
    print(f"Loading Hugging Face dataset '{hf_name}'...")
    try:
        dataset_dict = load_dataset(hf_name, cache_dir=cache_dir)
        
        if split not in dataset_dict:
            print(f"  [ERROR] Split '{split}' not found.")
            return []
        
        ds = dataset_dict[split]

        if image_class:
            print(f"  Filtering for class: '{image_class}'...")
            try:
                #map string "person" to its integer ID (usually 0)
                category_feature = ds.features["objects"].feature["category"]
                target_id = category_feature.str2int(image_class)
                
                #keep images where the target_id is in the list of categories
                ds = ds.filter(lambda x: target_id in x["objects"]["category"])
                print(f"  Filter complete. Found {len(ds)} images containing '{image_class}'.")
            except Exception as e:
                print(f"  [WARNING] Filtering failed: {e}. using full dataset.")

        #sample from the FILTERED dataset
        dataset_size = len(ds)
        if dataset_size == 0:
            print("  [ERROR] No images found after filtering.")
            return []

        if dataset_size > target_sample_size:
            print(f"  Sampling {target_sample_size} items from filtered results...")
            # We use .select() with random indices to get a new Dataset object
            indices = random.sample(range(dataset_size), target_sample_size)
            ds = ds.select(indices)
        
        #generate synthetic paths based on the final selection

        synthetic_data = [(f"{split}_{i}.jpg", i) for i in range(len(ds))]
        return synthetic_data

    except Exception as e:
        print(f"  [CRITICAL ERROR] Failed to load dataset: {e}")
        return []
    

def get_non_huggingface_dataset_paths(directory, target_sample_size):
    """
    Logic for the SAFE dataset folder structure. 
    Samples a specific number of images from each subdirectory (model).
    """
    all_files = []
    model_dirs = [d.path for d in os.scandir(directory) if d.is_dir()]
    if not model_dirs:
        print(f"  [WARNING] No model subdirectories found in {directory}")
        return []

    print(f"  Found {len(model_dirs)} model directories. Attempting to sample {target_sample_size} images from each.")

    for model_dir in sorted(model_dirs):
        model_image_files = []
        for ext in IMAGE_EXTENSIONS:
            model_image_files.extend(glob(os.path.join(model_dir, '**', f'*{ext}'), recursive=True))

        if not model_image_files:
            print(f"    - No images found in {os.path.basename(model_dir)}")
            continue
        if len(model_image_files) < target_sample_size:
            print(f"    - Found {len(model_image_files)} images in {os.path.basename(model_dir)} (less than target). Taking all.")
            all_files.extend(model_image_files)
        else:
            print(f"    - Sampling {target_sample_size} images from {os.path.basename(model_dir)}.")
            all_files.extend(random.sample(model_image_files, target_sample_size))
    return all_files

def get_standard_paths(directory):
    """Standard recursive glob search for image files."""
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob(os.path.join(directory, '**', f'*{ext}'), recursive=True))
    return files

# --- Helper Functions for Processing ---

def run_simulations_for_image(file_path, dataset_name, directory, simulator, csv_writer):
    """Runs all social media simulations for a single image and logs results."""
    try:
        media_info = get_media_info(file_path, dataset_name, directory)
        media_type, authenticity, original_filename, source_model = media_info
        
        simulations = {
            "facebook": lambda: simulator.facebook(file_path),
            "instagram_feed": lambda: simulator.instagram(file_path, post_type='feed'),
            "tiktok": lambda: simulator.tiktok(file_path),
            "whatsapp_standard": lambda: simulator.whatsapp(file_path, quality_mode='standard'),
            "signal_standard": lambda: simulator.signal(file_path, quality_setting='standard'),
            "telegram": lambda: simulator.telegram(file_path),
        }

        for sim_name, sim_func in simulations.items():
            try:
                sim_func() 
                platform_dir_name = sim_name.split('_')[0]
                output_dir = os.path.join(CURATED_DIR, platform_dir_name)
                
                # Setup paths
                processed_ext = ".jpg"
                temp_output_path = os.path.join(output_dir, f"TEMPOUT{processed_ext}")

                if os.path.exists(temp_output_path):
                    new_filename = f"{os.path.splitext(original_filename)[0]}_{sim_name}{processed_ext}"
                    new_filepath = os.path.join(output_dir, new_filename)
                    
                    shutil.move(temp_output_path, new_filepath)
                    csv_writer.writerow([
                        file_path, original_filename, media_type, authenticity, 
                        source_model, sim_name, new_filename, new_filepath
                    ])
            except Exception as e:
                print(f"  [ERROR] Simulation '{sim_name}' failed for {original_filename}: {e}")

    except Exception as e:
        print(f"  [ERROR] Metadata extraction failed for {os.path.basename(file_path)}: {e}")




def run_pipeline(dataset_name, target_sample_size=2000):
    print(f"--- Starting Image Curation Pipeline for {dataset_name} ---")

    # 1. Config Retrieval
    directory = DATASET_DIRS.get(dataset_name)
    storage_info = image_storage_type_dict.get(dataset_name, {})
    storage_type = storage_info.get("storage")
    
    if not directory:
        print(f"Error: Dataset '{dataset_name}' not found.")
        return

    # 2. Path/Data Discovery
    dataset_obj = None  # To keep HF dataset in memory if needed
    
    if storage_type == "hugging_face_lock_file":
        hf_name = storage_info.get("hf_name")
        # Now returns a list of (synthetic_name, index)
        all_files = get_hf_dataset_paths(hf_name, directory, target_sample_size)
        # We need the actual dataset object to pull the images in the loop
        from datasets import load_dataset
        dataset_obj = load_dataset(hf_name, cache_dir=directory)
    else:
        # Standard local file search
        if dataset_name == "SAFE":
            all_files = get_non_huggingface_dataset_paths(directory, target_sample_size)
        else:
            all_files = get_standard_paths(directory)

    if not all_files:
        print(f"No files found for {dataset_name}. Exiting.")
        return

    # 3. Setup
    metadata_path = os.path.join(CURATED_DIR, f"{dataset_name}_metadata.csv")
    simulator = SocialMediaSimulator(base_output_dir=CURATED_DIR)

    # 4. Processing Loop
    with open(metadata_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header each time to ensure the file is self-contained and fresh.
        csv_writer.writerow([
            'original_path', 'original_filename', 'media_type', 'authenticity',
            'source_model', 'simulation', 'processed_filename', 'processed_path'
        ])
        
        for item in tqdm(all_files, desc=f"Curating {dataset_name}"):
            try:
                if storage_type == "hugging_face_lock_file":
                    # --- HANDLING HUGGING FACE (In-Memory) ---
                    synthetic_name, idx = item
                    split = synthetic_name.split('_')[0] # Get 'val' from 'val_102.jpg'
                    
                    # Access the PIL image
                    pil_img = dataset_obj[split][idx]['image']
                    
                    # Create a temporary local file so the simulator can read it
                    temp_path = os.path.join(directory, f"TEMP_INPUT_{synthetic_name}")
                    pil_img.convert("RGB").save(temp_path) # Convert to RGB to ensure JPEG compatibility
                    
                    run_simulations_for_image(temp_path, dataset_name, directory, simulator, csv_writer)
                    
                    # Cleanup the temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                else:
                    # --- HANDLING LOCAL STORAGE ---
                    # item is a direct file_path (string)
                    run_simulations_for_image(item, dataset_name, directory, simulator, csv_writer)

            except Exception as e:
                print(f"  [CRITICAL ERROR] Failed to process {item}: {e}")

    print(f"--- {dataset_name} Curation Finished ---")