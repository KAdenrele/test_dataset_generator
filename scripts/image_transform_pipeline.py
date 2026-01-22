import os
import shutil
from glob import glob
from tqdm import tqdm
import csv
import random
from datasets import load_dataset
from scripts.media_processes import SocialMediaSimulator
import logging

random.seed(42)  

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.tiff', '.heic']

ALL_SIMULATIONS = [
    "original", "whatsapp_document", "signal_document","telegram_document",
    "facebook", "instagram_feed", "instagram_story", "instagram_reel", "tiktok",
    "whatsapp_standard_media", "whatsapp_high_media", 
    "signal_standard_media", "signal_high_media", 
    "telegram_media", 
]


def get_media_info(file_path, dataset_name, base_dir):
    """Extracts metadata from the file path and dataset info."""
    media_type = "image"
    original_filename = os.path.basename(file_path)

    source_model = None
    source_model_details = None
    if dataset_name == "SAFE":
        try:
            relative_dir_path = os.path.relpath(os.path.dirname(file_path), base_dir)
            # The model name is the first component of this relative path... he says... hopefully
            path_components = relative_dir_path.split(os.sep)
            source_model = path_components[0]
            if len(path_components) > 1:
                source_model_details = os.path.join(*path_components[1:])
        except (ValueError, IndexError):
            source_model = "unknown" # Should not happen if paths are correct
            source_model_details = "unknown"
    
    return media_type, original_filename, source_model, source_model_details

def get_hf_dataset_paths(hf_name, cache_dir, target_sample_size, split='val', image_class="person"):
    """
    Loads a HF dataset, finds original indices for a specific class, samples them,
    and returns a list of (synthetic_path, original_index) tuples.
    """
    logging.info(f"Loading and processing Hugging Face dataset '{hf_name}'...")
    try:
        # Load the dataset dictionary
        dataset_dict = load_dataset(hf_name, cache_dir=cache_dir)
        if split not in dataset_dict:
            logging.error(f"Split '{split}' not found. Available: {list(dataset_dict.keys())}")
            return []
        
        ds = dataset_dict[split]
        logging.info(f"Original size of '{split}' split: {len(ds)} images.")

        valid_indices = []
        if image_class:
            logging.info(f"Filtering for class: '{image_class}'...")
            try:
                # Correctly access the ClassLabel feature to get the integer ID for the class
                category_feature = ds.features["objects"]["category"].feature
                target_id = category_feature.str2int(image_class)

                # Efficiently iterate over just the category column to find matching indices
                logging.info("Scanning for matching images (this may take a moment)...")
                for i, categories in enumerate(tqdm(ds['objects']['category'], desc="Filtering")):
                    if target_id in categories:
                        valid_indices.append(i)
                logging.info(f"Found {len(valid_indices)} images containing '{image_class}'.")

            except (KeyError, AttributeError, ValueError) as e:
                logging.warning(f"Could not filter by class '{image_class}'. The dataset might not have the expected structure or class. Error: {e}")
            except Exception as e:
                logging.warning(f"An unexpected error occurred during filtering: {e}")

        # If filtering was skipped, failed, or found nothing, use all original indices
        if not valid_indices:
            logging.info("No valid images found after filtering, or filtering was skipped. Using all images.")
            valid_indices = list(range(len(ds)))

        # Sample from the list of valid original indices
        if len(valid_indices) > target_sample_size:
            logging.info(f"Sampling {target_sample_size} from {len(valid_indices)} valid images...")
            sampled_indices = random.sample(valid_indices, target_sample_size)
        else:
            logging.info(f"Using all {len(valid_indices)} valid images.")
            sampled_indices = valid_indices
        
        # Return list of (synthetic_name, original_index)
        logging.info(f"Returning {len(sampled_indices)} indices for processing.")
        return [(f"{split}_{idx}.jpg", idx) for idx in sampled_indices]

    except Exception as e:
        logging.error(f"Failed to load or process dataset '{hf_name}'. Reason: {e}")
        return []
    

def get_non_huggingface_dataset_paths(directory, target_sample_size):
    """
    Logic for the SAFE dataset folder structure. 
    Samples a specific number of images from each subdirectory (model).
    """
    all_files = []
    model_dirs = [d.path for d in os.scandir(directory) if d.is_dir()]
    if not model_dirs:
        logging.warning(f"No model subdirectories found in {directory}")
        return []

    logging.info(f"Found {len(model_dirs)} model directories. Attempting to sample {target_sample_size} images from each.")

    for model_dir in sorted(model_dirs):
        model_image_files = []
        for ext in IMAGE_EXTENSIONS:
            model_image_files.extend(glob(os.path.join(model_dir, '**', f'*{ext}'), recursive=True))

        if not model_image_files:
            logging.warning(f"No images found in {os.path.basename(model_dir)}")
            continue
        if len(model_image_files) < target_sample_size:
            logging.info(f"Found {len(model_image_files)} images in {os.path.basename(model_dir)} (less than target). Taking all.")
            all_files.extend(model_image_files)
        else:
            logging.info(f"Sampling {target_sample_size} images from {os.path.basename(model_dir)}.")
            all_files.extend(random.sample(model_image_files, target_sample_size))
    return all_files

def get_standard_paths(directory):
    """Standard recursive glob search for image files."""
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob(os.path.join(directory, '**', f'*{ext}'), recursive=True))
    return files


def run_simulations_for_image(file_path, dataset_name, directory, simulator, csv_writer, authenticity, simulations_to_run, curated_dir, originals_dir):
    """Runs all social media simulations for a single image and logs results."""
    try:
        media_type, original_filename, source_model, source_model_details = get_media_info(file_path, dataset_name, directory)

        # Create a unique base filename from the relative path to prevent overwrites.
        # This is crucial for datasets that have identical filenames in different subdirectories.
        try:
            # e.g., 'model_A/subdir/image.png' -> 'model_A_subdir_image'
            relative_path = os.path.relpath(file_path, directory)
            unique_base = os.path.splitext(relative_path)[0].replace(os.sep, '_')
        except ValueError:
            # Fallback for cases where relpath fails or for temporary files from Hugging Face datasets.
            unique_base = os.path.splitext(original_filename)[0]

        if "original" in simulations_to_run:
            # --- Save the original unprocessed image and log it ---
            try:
                _, original_ext = os.path.splitext(original_filename)
                original_save_filename = f"{unique_base}_original{original_ext}"
                original_save_path = os.path.join(originals_dir, original_save_filename)
                if os.path.exists(original_save_path):
                    logging.info(f"Skipping existing original file: {original_save_path}")
                else:
                    shutil.copy2(file_path, original_save_path)

                    # Prepare and write the row for the original image to the CSV.
                    base_row_data = [file_path, original_filename, media_type, authenticity, source_model, source_model_details, original_save_filename, original_save_path]
                    one_hot_sims = [1 if sim == "original" else 0 for sim in ALL_SIMULATIONS]
                    csv_writer.writerow(base_row_data + one_hot_sims)
            except Exception as e:
                logging.warning(f"Could not save or log original file {original_filename}: {e}")

        all_simulations_map = {
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

        for sim_name in simulations_to_run:
            if sim_name == "original":
                continue # Already handled
            sim_func = all_simulations_map.get(sim_name)
            if not sim_func:
                logging.warning(f"Unknown simulation '{sim_name}' requested. Skipping.")
                continue
            try:
                _, original_ext = os.path.splitext(original_filename)
                # Most simulations convert to JPG, but 'document' types preserve the original file extension.
                processed_ext = original_ext if 'document' in sim_name else ".jpg"
                output_dir = os.path.join(curated_dir, sim_name)
                new_filename = f"{unique_base}_{sim_name}{processed_ext}"
                new_filepath = os.path.join(output_dir, new_filename)

                if os.path.exists(new_filepath):
                    logging.info(f"Skipping existing simulation file: {new_filepath}")
                    continue

                sim_func()

                # The simulator API saves output to a platform-specific subdirectory (e.g., 'facebook', 'instagram').
                platform_dir_name = sim_name.split('_')[0]
                temp_output_path = os.path.join(simulator.base_output_dir, platform_dir_name, f"TEMPOUT{processed_ext}")
                if os.path.exists(temp_output_path):
                    # Define the final, specific directory for this simulation (e.g., "instagram_story").
                    os.makedirs(output_dir, exist_ok=True)

                    # Move the processed file from the temp location to its final destination.
                    shutil.move(temp_output_path, new_filepath)

                    # Prepare and write the single, one-hot encoded row to the CSV.
                    base_row_data = [file_path, original_filename, media_type, authenticity, source_model, source_model_details, new_filename, new_filepath]
                    one_hot_sims = [1 if sim == sim_name else 0 for sim in ALL_SIMULATIONS]
                    csv_writer.writerow(base_row_data + one_hot_sims)
            except Exception as e:
                logging.error(f"Simulation '{sim_name}' failed for {original_filename}: {e}")

    except Exception as e:
        logging.error(f"Metadata extraction failed for {os.path.basename(file_path)}: {e}")




def run_pipeline(
    dataset_name: str,
    image_directory_path: str,
    destination_directory: str,
    is_huggingface: bool,
    has_subdirectories: bool,
    is_synthetic: bool,
    simulations_to_run: list,
    hf_name: str = None,
    target_sample_size: int = 2000
):
    logging.info(f"--- Starting Image Curation Pipeline for {dataset_name} ---")

    # 1. Setup destination directories
    CURATED_DIR = destination_directory
    os.makedirs(CURATED_DIR, exist_ok=True)
    ORIGINALS_DIR = os.path.join(CURATED_DIR, "originals")
    os.makedirs(ORIGINALS_DIR, exist_ok=True)

    directory = image_directory_path
    authenticity = "synthetic" if is_synthetic else "authentic"

    # 2. Path/Data Discovery
    dataset_obj = None  # To keep HF dataset in memory if needed
    
    if is_huggingface:
        if not hf_name:
            logging.error(f"Hugging Face dataset name (hf_name) must be provided for {dataset_name}.")
            return
        # Now returns a list of (synthetic_name, index)
        all_files = get_hf_dataset_paths(hf_name, directory, target_sample_size)
        # We need the actual dataset object to pull the images in the loop
        from datasets import load_dataset
        dataset_obj = load_dataset(hf_name, cache_dir=directory)
    else:
        # Standard local file search
        if has_subdirectories:
            all_files = get_non_huggingface_dataset_paths(directory, target_sample_size)
        else:
            all_files = get_standard_paths(directory)

    if not all_files:
        logging.info(f"No files found for {dataset_name}. Exiting.")
        return

    # 3. Setup
    metadata_path = os.path.join(CURATED_DIR, f"{dataset_name}_metadata.csv")
    write_header = not os.path.exists(metadata_path)
    simulator = SocialMediaSimulator(base_output_dir=CURATED_DIR)

    # 4. Processing Loop
    with open(metadata_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        if write_header:
            header = [
                'original_path', 'original_filename', 'media_type', 'authenticity',
                'source_model', 'source_model_details', 'processed_filename', 'processed_path'
            ]
            # Add the one-hot encoded simulation columns to the header.
            header.extend(ALL_SIMULATIONS)
            csv_writer.writerow(header)
        
        for item in tqdm(all_files, desc=f"Curating {dataset_name}"):
            try:
                if is_huggingface:
                    # --- HANDLING HUGGING FACE (In-Memory) ---
                    synthetic_name, idx = item
                    split = synthetic_name.split('_')[0] # Get 'val' from 'val_102.jpg'
                    
                    # Access the PIL image
                    pil_img = dataset_obj[split][idx]['image']
                    
                    # Create a temporary local file so the simulator can read it
                    temp_path = os.path.join(directory, f"TEMP_INPUT_{synthetic_name}")
                    pil_img.convert("RGB").save(temp_path) # Convert to RGB to ensure JPEG compatibility
                    
                    run_simulations_for_image(temp_path, dataset_name, directory, simulator, csv_writer, authenticity, simulations_to_run, CURATED_DIR, ORIGINALS_DIR)
                    
                    # Cleanup the temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                else:
                    # --- HANDLING LOCAL STORAGE ---
                    # item is a direct file_path (string)
                    run_simulations_for_image(item, dataset_name, directory, simulator, csv_writer, authenticity, simulations_to_run, CURATED_DIR, ORIGINALS_DIR)

            except Exception as e:
                logging.error(f"Failed to process {item}: {e}")

    # 5. Cleanup
    # The simulator API creates base directories (e.g., 'instagram') to store temp files.
    # After the pipeline moves these files to their final specific directories (e.g., 'instagram_story'),
    # these base directories are left empty. This step removes them.
    logging.info("Cleaning up empty intermediate directories...")
    platforms_with_subtypes = ["instagram", "whatsapp", "signal", "telegram"]
    for platform in platforms_with_subtypes:
        intermediate_dir_path = os.path.join(CURATED_DIR, platform)
        try:
            if os.path.isdir(intermediate_dir_path) and not os.listdir(intermediate_dir_path):
                os.rmdir(intermediate_dir_path)
                logging.info(f"Removed empty directory: {platform}")
        except OSError as e:
            logging.warning(f"Could not remove directory {intermediate_dir_path}: {e}")

    logging.info(f"--- {dataset_name} Curation Finished ---")