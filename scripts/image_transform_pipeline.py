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
    #"SAFE": os.path.join(BASE_DIR, "raw/safe_images_synthetic"),
    "SAFE": os.path.join(BASE_DIR, "SAFEDataset/data"),
    "COCO": os.path.join(BASE_DIR, "raw/coco_images_authentic")
}

CURATED_DIR = os.path.join(BASE_DIR, "curated/images")
#remove the curated directory if it exists from a previous execution.
if os.path.exists(CURATED_DIR):
    shutil.rmtree(CURATED_DIR)
os.makedirs(CURATED_DIR)

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.tiff', '.heic']

ALL_SIMULATIONS = [
    "facebook", "instagram_feed", "instagram_story", "instagram_reel", "tiktok",
    "whatsapp_standard_media", "whatsapp_high_media", "whatsapp_document",
    "signal_standard_media", "signal_high_media", "signal_document",
    "telegram_media", "telegram_document",
]


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
    """
    Loads a HF dataset, finds original indices for a specific class, samples them,
    and returns a list of (synthetic_path, original_index) tuples.
    """
    print(f"Loading and processing Hugging Face dataset '{hf_name}'...")
    try:
        # Load the dataset dictionary
        dataset_dict = load_dataset(hf_name, cache_dir=cache_dir)
        if split not in dataset_dict:
            print(f"  [ERROR] Split '{split}' not found. Available: {list(dataset_dict.keys())}")
            return []
        
        ds = dataset_dict[split]
        print(f"  Original size of '{split}' split: {len(ds)} images.")

        valid_indices = []
        if image_class:
            print(f"  Filtering for class: '{image_class}'...")
            try:
                # Correctly access the ClassLabel feature to get the integer ID for the class
                category_feature = ds.features["objects"]["category"].feature
                target_id = category_feature.str2int(image_class)

                # Efficiently iterate over just the category column to find matching indices
                print("  Scanning for matching images (this may take a moment)...")
                for i, categories in enumerate(tqdm(ds['objects']['category'], desc="Filtering")):
                    if target_id in categories:
                        valid_indices.append(i)
                print(f"  Found {len(valid_indices)} images containing '{image_class}'.")

            except (KeyError, AttributeError, ValueError) as e:
                print(f"  [WARNING] Could not filter by class '{image_class}'. The dataset might not have the expected structure or class. Error: {e}")
            except Exception as e:
                print(f"  [WARNING] An unexpected error occurred during filtering: {e}")

        # If filtering was skipped, failed, or found nothing, use all original indices
        if not valid_indices:
            print("  No valid images found after filtering, or filtering was skipped. Using all images.")
            valid_indices = list(range(len(ds)))

        # Sample from the list of valid original indices
        if len(valid_indices) > target_sample_size:
            print(f"  Sampling {target_sample_size} from {len(valid_indices)} valid images...")
            sampled_indices = random.sample(valid_indices, target_sample_size)
        else:
            print(f"  Using all {len(valid_indices)} valid images.")
            sampled_indices = valid_indices
        
        # Return list of (synthetic_name, original_index)
        print(f"  Returning {len(sampled_indices)} indices for processing.")
        return [(f"{split}_{idx}.jpg", idx) for idx in sampled_indices]

    except Exception as e:
        print(f"  [CRITICAL ERROR] Failed to load or process dataset '{hf_name}'. Reason: {e}")
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


def run_simulations_for_image(file_path, dataset_name, directory, simulator, csv_writer):
    """Runs all social media simulations for a single image and logs results."""
    try:
        media_info = get_media_info(file_path, dataset_name, directory)
        media_type, authenticity, original_filename, source_model = media_info

        # Create a unique base filename from the relative path to prevent overwrites.
        # This is crucial for datasets that have identical filenames in different subdirectories.
        try:
            # e.g., 'model_A/subdir/image.png' -> 'model_A_subdir_image'
            relative_path = os.path.relpath(file_path, directory)
            unique_base = os.path.splitext(relative_path)[0].replace(os.sep, '_')
        except ValueError:
            # Fallback for cases where relpath fails or for temporary files from Hugging Face datasets.
            unique_base = os.path.splitext(original_filename)[0]

        simulations = {
            "facebook": lambda: simulator.facebook(file_path),
            # Instagram
            "instagram_feed": lambda: simulator.instagram(file_path, post_type='feed'),
            "instagram_story": lambda: simulator.instagram(file_path, post_type='story'),
            "instagram_reel": lambda: simulator.instagram(file_path, post_type='reel'),
            # TikTok
            "tiktok": lambda: simulator.tiktok(file_path),
            # WhatsApp
            "whatsapp_standard_media": lambda: simulator.whatsapp(file_path, quality_mode='standard', upload_type='media'),
            "whatsapp_high_media": lambda: simulator.whatsapp(file_path, quality_mode='high', upload_type='media'),
            "whatsapp_document": lambda: simulator.whatsapp(file_path, upload_type='document'),
            # Signal
            "signal_standard_media": lambda: simulator.signal(file_path, quality_setting='standard', as_document=False),
            "signal_high_media": lambda: simulator.signal(file_path, quality_setting='high', as_document=False),
            "signal_document": lambda: simulator.signal(file_path, as_document=True),
            # Telegram
            "telegram_media": lambda: simulator.telegram(file_path, as_document=False),
            "telegram_document": lambda: simulator.telegram(file_path, as_document=True),
        }

        for sim_name, sim_func in simulations.items():
            try:
                sim_func()

                _, original_ext = os.path.splitext(original_filename)

                # Most simulations convert to JPG, but 'document' types preserve the original file extension.
                processed_ext = original_ext if 'document' in sim_name else ".jpg"

                # The simulator API saves output to a platform-specific subdirectory (e.g., 'facebook', 'instagram').
                platform_dir_name = sim_name.split('_')[0]
                temp_output_path = os.path.join(simulator.base_output_dir, platform_dir_name, f"TEMPOUT{processed_ext}")
                if os.path.exists(temp_output_path):
                    # Define the final, specific directory for this simulation (e.g., "instagram_story").
                    output_dir = os.path.join(CURATED_DIR, sim_name)
                    os.makedirs(output_dir, exist_ok=True)

                    # Define the final filename and path.
                    new_filename = f"{unique_base}_{sim_name}{processed_ext}"
                    new_filepath = os.path.join(output_dir, new_filename)

                    # Move the processed file from the temp location to its final destination.
                    shutil.move(temp_output_path, new_filepath)

                    # Prepare and write the single, one-hot encoded row to the CSV.
                    base_row_data = [file_path, original_filename, media_type, authenticity, source_model, new_filename, new_filepath]
                    one_hot_sims = [1 if sim == sim_name else 0 for sim in ALL_SIMULATIONS]
                    csv_writer.writerow(base_row_data + one_hot_sims)
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
        header = [
            'original_path', 'original_filename', 'media_type', 'authenticity',
            'source_model', 'processed_filename', 'processed_path'
        ]
        # Add the one-hot encoded simulation columns to the header.
        header.extend(ALL_SIMULATIONS)
        csv_writer.writerow(header)
        
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

    # 5. Cleanup
    # The simulator API creates base directories (e.g., 'instagram') to store temp files.
    # After the pipeline moves these files to their final specific directories (e.g., 'instagram_story'),
    # these base directories are left empty. This step removes them.
    print("  Cleaning up empty intermediate directories...")
    platforms_with_subtypes = ["instagram", "whatsapp", "signal", "telegram"]
    for platform in platforms_with_subtypes:
        intermediate_dir_path = os.path.join(CURATED_DIR, platform)
        try:
            if os.path.isdir(intermediate_dir_path) and not os.listdir(intermediate_dir_path):
                os.rmdir(intermediate_dir_path)
                print(f"    - Removed empty directory: {platform}")
        except OSError as e:
            print(f"  [WARNING] Could not remove directory {intermediate_dir_path}: {e}")

    print(f"--- {dataset_name} Curation Finished ---")