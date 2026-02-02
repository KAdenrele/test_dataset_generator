import os
import shutil
from glob import glob
from tqdm import tqdm
import csv
import random
from datasets import load_dataset
from scripts.media_processes import SocialMediaSimulator
import concurrent.futures
from datasets import get_dataset_infos
import logging

random.seed(42)  

VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm']

ALL_SIMULATIONS = [
    "whatsapp_document", "signal_document","telegram_document","original", 
    "facebook", "instagram_feed", "instagram_story", "instagram_reel", "tiktok",
    "whatsapp_standard_media", "whatsapp_high_media", 
    "signal_standard_media", "signal_high_media", 
    "telegram_media", 
]


def get_media_info(file_path, dataset_name, base_dir):
    """Extracts metadata from the file path and dataset info."""
    media_type = "video"
    original_filename = os.path.basename(file_path)

    source_model = None
    source_model_details = None
    if dataset_name == "SAFE":
        try:
            relative_dir_path = os.path.relpath(os.path.dirname(file_path), base_dir)
            # The model name is the first component of this relative path.
            path_components = relative_dir_path.split(os.sep)
            source_model = path_components[0]
            if len(path_components) > 1:
                source_model_details = os.path.join(*path_components[1:])
        except (ValueError, IndexError):
            source_model = "unknown" # Should not happen if paths are correct
            source_model_details = "unknown"
    
    return media_type, original_filename, source_model, source_model_details

def get_hf_video_class_names(hf_name: str) -> list:
    """
    Loads a Hugging Face dataset's metadata and extracts the class names.

    This function fetches metadata without downloading the full dataset, making it efficient
    for discovering class labels.

    Args:
        hf_name: The name of the Hugging Face dataset (e.g., "ucf101").

    Returns:
        A list of class name strings, or an empty list if they cannot be found.
    """
    logging.info(f"Discovering class names from '{hf_name}'...")
    try:
        # Fetches all configuration metadata for the dataset.
        infos = get_dataset_infos(hf_name)
        # Assume the first configuration is the desired one.
        config_name = list(infos.keys())[0]
        features = infos[config_name].features

        if "label" in features and hasattr(features["label"], 'names'):
            class_names = features["label"].names
            logging.info(f"Discovered {len(class_names)} classes.")
            return class_names
        else:
            logging.warning("Could not automatically determine class names. The 'label' feature might be missing or not a ClassLabel type.")
            return []
    except Exception as e:
        logging.error(f"Failed to fetch dataset info for '{hf_name}'. Reason: {e}")
        return []

def get_hf_video_dataset_paths(hf_name, cache_dir, target_sample_size, split='train', video_class: str = None):
    """
    Loads a HF video dataset, samples it, and returns a list of file paths.
    If `video_class` is specified, it will filter the dataset for that class before sampling.
    """
    logging.info(f"Loading and processing Hugging Face dataset '{hf_name}'...")
    try:
        dataset_dict = load_dataset(hf_name, cache_dir=cache_dir)
        if split not in dataset_dict:
            logging.error(f"Split '{split}' not found. Available: {list(dataset_dict.keys())}")
            return []

        ds = dataset_dict[split]
        logging.info(f"Original size of '{split}' split: {len(ds)} videos.")

        indices_to_sample_from = list(range(len(ds)))

        if video_class:
            logging.info(f"Filtering for class: '{video_class}'...")
            try:
                label_feature = ds.features["label"]
                target_id = label_feature.str2int(video_class)
                indices_to_sample_from = [i for i, label in enumerate(tqdm(ds['label'], desc=f"Filtering for {video_class}")) if label == target_id]
                logging.info(f"Found {len(indices_to_sample_from)} videos for class '{video_class}'.")
            except (KeyError, ValueError, AttributeError) as e:
                logging.warning(f"Could not filter by class '{video_class}'. The dataset may lack a 'label' feature or the class name is incorrect. Error: {e}. Proceeding with all videos.")

        if len(indices_to_sample_from) > target_sample_size:
            logging.info(f"Sampling {target_sample_size} from {len(indices_to_sample_from)} videos...")
            indices = random.sample(indices_to_sample_from, target_sample_size)
        else:
            logging.info(f"Using all {len(indices_to_sample_from)} videos.")
            indices = indices_to_sample_from

        video_paths = []
        logging.info(f"Extracting file paths for {len(indices)} sampled videos...")
        for i in tqdm(indices, desc="Extracting video paths"):
            item = ds[i]
            # This assumes the dataset yields dictionaries with a 'video' key,
            # which in turn is a dictionary containing a 'path' to the file.
            if 'video' in item and isinstance(item['video'], dict) and 'path' in item['video']:
                video_path = item['video']['path']
                if video_path and os.path.exists(video_path):
                    video_paths.append(video_path)
                else:
                    logging.warning(f"Video path not found or does not exist for index {i}: {video_path}")
            else:
                logging.warning(f"Could not find a 'video' object with a 'path' for index {i}. Item keys: {item.keys()}")

        logging.info(f"Returning {len(video_paths)} video paths for processing.")
        return video_paths

    except Exception as e:
        logging.error(f"Failed to load or process dataset '{hf_name}'. Reason: {e}")
        return []

def get_non_huggingface_video_dataset_paths(directory, target_sample_size):
    """
    Logic for a dataset folder structure with subdirectories. 
    Samples a specific number of videos from each subdirectory.
    """
    all_files = []
    model_dirs = [d.path for d in os.scandir(directory) if d.is_dir()]
    if not model_dirs:
        logging.warning(f"No model subdirectories found in {directory}")
        return []

    logging.info(f"Found {len(model_dirs)} model directories. Attempting to sample {target_sample_size} videos from each.")

    for model_dir in sorted(model_dirs):
        model_video_files = []
        for ext in VIDEO_EXTENSIONS:
            model_video_files.extend(glob(os.path.join(model_dir, '**', f'*{ext}'), recursive=True))

        if not model_video_files:
            logging.warning(f"No videos found in {os.path.basename(model_dir)}")
            continue
        if len(model_video_files) < target_sample_size:
            logging.info(f"Found {len(model_video_files)} videos in {os.path.basename(model_dir)} (less than target). Taking all.")
            all_files.extend(model_video_files)
        else:
            logging.info(f"Sampling {target_sample_size} videos from {os.path.basename(model_dir)}.")
            all_files.extend(random.sample(model_video_files, target_sample_size))
    return all_files

def get_standard_video_paths(directory):
    """Standard recursive glob search for video files."""
    files = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(glob(os.path.join(directory, '**', f'*{ext}'), recursive=True))
    return files


def run_simulations_for_video(file_path, dataset_name, directory, simulator, authenticity, simulations_to_run, curated_dir, originals_dir):
    """Runs all social media simulations for a single video and logs results."""
    rows_to_write = []
    try:
        media_type, original_filename, source_model, source_model_details = get_media_info(file_path, dataset_name, directory)

        try:
            relative_path = os.path.relpath(file_path, directory)
            unique_base = os.path.splitext(relative_path)[0].replace(os.sep, '_')
        except ValueError:
            unique_base = os.path.splitext(original_filename)[0]

        if "original" in simulations_to_run:
            try:
                _, original_ext = os.path.splitext(original_filename)
                original_save_filename = f"{unique_base}_original{original_ext}"
                original_save_path = os.path.join(originals_dir, original_save_filename)
                if os.path.exists(original_save_path):
                    logging.info(f"Skipping existing original file: {original_save_path}")
                else:
                    shutil.copy2(file_path, original_save_path)

                base_row_data = [file_path, original_filename, media_type, authenticity, source_model, source_model_details, original_save_filename, original_save_path]
                one_hot_sims = [1 if sim == "original" else 0 for sim in ALL_SIMULATIONS]
                rows_to_write.append(base_row_data + one_hot_sims)
            except Exception as e:
                logging.warning(f"Could not save or log original file {original_filename}: {e}")

        all_simulations_map = {
            "facebook": lambda: simulator.facebook(file_path),
            "instagram_feed": lambda: simulator.instagram(file_path, post_type='feed'),
            "instagram_story": lambda: simulator.instagram(file_path, post_type='story'),
            "instagram_reel": lambda: simulator.instagram(file_path, post_type='reel'),
            "tiktok": lambda: simulator.tiktok(file_path),
            "whatsapp_standard_media": lambda: simulator.whatsapp(file_path, quality_mode='standard', upload_type='media'),
            "whatsapp_high_media": lambda: simulator.whatsapp(file_path, quality_mode='high', upload_type='media'),
            "whatsapp_document": lambda: simulator.whatsapp(file_path, upload_type='document'),
            "signal_standard_media": lambda: simulator.signal(file_path, quality_setting='standard', as_document=False),
            "signal_high_media": lambda: simulator.signal(file_path, quality_setting='high', as_document=False),
            "signal_document": lambda: simulator.signal(file_path, as_document=True),
            "telegram_media": lambda: simulator.telegram(file_path, as_document=False),
            "telegram_document": lambda: simulator.telegram(file_path, as_document=True),
        }

        for sim_name in simulations_to_run:
            if sim_name == "original":
                continue
            sim_func = all_simulations_map.get(sim_name)
            if not sim_func:
                logging.warning(f"Unknown simulation '{sim_name}' requested. Skipping.")
                continue
            try:
                _, original_ext = os.path.splitext(original_filename)
                processed_ext = original_ext if 'document' in sim_name else ".mp4"
                output_dir = os.path.join(curated_dir, sim_name)
                new_filename = f"{unique_base}_{sim_name}{processed_ext}"
                new_filepath = os.path.join(output_dir, new_filename)

                if not os.path.exists(new_filepath):
                    sim_func()
                    platform_dir_name = sim_name.split('_')[0]
                    temp_output_path = os.path.join(simulator.base_output_dir, platform_dir_name, f"TEMPOUT{processed_ext}")
                    if os.path.exists(temp_output_path):
                        os.makedirs(output_dir, exist_ok=True)
                        shutil.move(temp_output_path, new_filepath)
                    else:
                        logging.warning(f"Simulation '{sim_name}' did not produce an output file. Skipping log entry.")
                        continue
                else:
                    logging.info(f"Skipping existing simulation file: {new_filepath}")

                base_row_data = [file_path, original_filename, media_type, authenticity, source_model, source_model_details, new_filename, new_filepath]
                one_hot_sims = [1 if sim == sim_name else 0 for sim in ALL_SIMULATIONS]
                rows_to_write.append(base_row_data + one_hot_sims)
            except Exception as e:
                logging.error(f"Simulation '{sim_name}' failed for {original_filename}: {e}")

    except Exception as e:
        logging.error(f"Metadata extraction failed for {os.path.basename(file_path)}: {e}")
    
    return rows_to_write


def _process_item_worker(args):
    """
    A picklable, top-level worker function for parallel processing.
    """
    (file_path, dataset_name, base_directory, destination_directory, 
     authenticity, simulations_to_run) = args

    curated_dir = destination_directory
    
    worker_temp_dir = os.path.join(curated_dir, f"worker_temp_{os.getpid()}_{random.randint(0, 999999)}")
    
    try:
        originals_dir = os.path.join(curated_dir, "originals")
        simulator = SocialMediaSimulator(base_output_dir=worker_temp_dir)
        
        return run_simulations_for_video(
            file_path, dataset_name, base_directory, simulator, 
            authenticity, simulations_to_run, curated_dir, originals_dir
        )
    finally:
        if os.path.exists(worker_temp_dir):
            shutil.rmtree(worker_temp_dir)

def run_pipeline(
    dataset_name: str,
    video_directory_path: str,
    destination_directory: str,
    is_huggingface: bool,
    has_subdirectories: bool,
    is_synthetic: bool,
    simulations_to_run: list,
    hf_name: str = None,
    video_class: str = None,
    target_sample_size: int = 2000,
    max_workers: int = None
):
    logging.info(f"--- Starting Video Curation Pipeline for {dataset_name} ---")

    CURATED_DIR = destination_directory
    os.makedirs(CURATED_DIR, exist_ok=True)
    ORIGINALS_DIR = os.path.join(CURATED_DIR, "originals")
    os.makedirs(ORIGINALS_DIR, exist_ok=True)

    directory = video_directory_path
    authenticity = "synthetic" if is_synthetic else "authentic"

    files_to_process = []
    base_directory_for_relpath = directory

    if is_huggingface:
        if not hf_name:
            logging.error(f"Hugging Face dataset name (hf_name) must be provided for {dataset_name}.")
            return
        
        logging.info("Loading Hugging Face dataset and extracting video paths...")
        files_to_process = get_hf_video_dataset_paths(hf_name, directory, target_sample_size, split='train', video_class=video_class)
        
        base_directory_for_relpath = directory

    else:
        if has_subdirectories:
            files_to_process = get_non_huggingface_video_dataset_paths(directory, target_sample_size)
        else:
            files_to_process = get_standard_video_paths(directory)

    if not files_to_process:
        logging.info(f"No files found for processing. Exiting.")
        return

    metadata_path = os.path.join(CURATED_DIR, f"{dataset_name}_metadata.csv")
    write_header = not os.path.exists(metadata_path)

    tasks = [
        (file_path, dataset_name, base_directory_for_relpath, destination_directory, 
         authenticity, simulations_to_run) 
        for file_path in files_to_process
    ]

    all_rows = []

    num_workers = max_workers
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, cpu_count // 2)

    logging.info(f"Starting parallel processing with {num_workers} workers.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = tqdm(executor.map(_process_item_worker, tasks), total=len(tasks), desc=f"Curating {dataset_name}")
        for result_rows in results:
            if result_rows:
                all_rows.extend(result_rows)

    with open(metadata_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        if write_header:
            header = [
                'original_path', 'original_filename', 'media_type', 'authenticity',
                'source_model', 'source_model_details', 'processed_filename', 'processed_path'
            ]
            header.extend(ALL_SIMULATIONS)
            csv_writer.writerow(header)
        if all_rows:
            csv_writer.writerows(all_rows)

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