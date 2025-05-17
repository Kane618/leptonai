import os
import json
import yaml
import uuid
import math
import shutil
import fire
import torch
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tianshou.data import Batch
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Dict, List, Any, Tuple
import subprocess
import multiprocessing as mp
from multiprocessing import Pool
from diffsynth import ModelManager, WanVideoPipeline, save_video, load_state_dict, get_lora_loaders
from huggingface_hub import snapshot_download
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.idm import IDM
from gr00t.utils.video import get_all_frames_and_timestamps
from gr00t.data.embodiment_tags import EmbodimentTag
from loguru import logger

# compress lerobot dataset
def compress_directory_to_zip(directory_path, output_zip_path):
    shutil.make_archive(output_zip_path, 'zip', directory_path)

# dump_idm_action
def load_dataset_and_config(checkpoint_path, validation_dataset_path, video_indices):
    config_path = os.path.join(checkpoint_path, "experiment_cfg", "conf.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    cfg = OmegaConf.create(config)

    dataset_name = os.path.basename(validation_dataset_path).split(".")[0]
    embodiment = dataset_name
    logger.info(f"Dataset name: {dataset_name}")

    modality_configs = cfg["modality_configs"][embodiment]
    if video_indices is not None:
        video_delta_indices = video_indices.split(" ")
        logger.info(f"Using provided video_delta_indices: {video_delta_indices}")
        modality_configs["video"]["delta_indices"] = video_delta_indices
    modality_configs = instantiate(modality_configs)


    if "all_transforms" in cfg:
        transform = cfg["all_transforms"][embodiment]
    else:
        transform = cfg["train_dataset"]["all_transforms"][embodiment]

    # Filter out VideoColorJitter transform for inference
    if "transforms" in transform:
        filtered_transforms = []
        for t in transform["transforms"]:
            if t.get("_target_") != "groot.data.transform.VideoColorJitter":
                filtered_transforms.append(t)

        transform["transforms"] = filtered_transforms
    transform_inst = instantiate(transform)

    metadata_versions = cfg["metadata_versions"]
    metadata_version = metadata_versions[embodiment]

    if "gr1" in embodiment:
        embodiment_tag = EmbodimentTag.GR1_unified
    elif "franka" in embodiment:
        embodiment_tag = EmbodimentTag.FRANKA
    elif "so100" in embodiment:
        embodiment_tag = EmbodimentTag.SO100
    elif "robocasa" in embodiment:
        embodiment_tag = EmbodimentTag.ROBOCASA
    else:
        raise ValueError(f"Unknown embodiment: {embodiment}")

    dataset = LeRobotSingleDataset(
        dataset_path=validation_dataset_path,
        modality_configs=modality_configs,
        # metadata_version=metadata_version,
        transforms=transform_inst,
        embodiment_tag=embodiment_tag,
    )

    return cfg, dataset, modality_configs


def collate_fn(features_list, device):
    batch_dict = {}
    keys = features_list[0].keys()
    for key in keys:
        if key in ["images", "view_ids"]:
            vals = [f[key] for f in features_list]
            batch_dict[key] = torch.as_tensor(np.concatenate(vals), device=device)
        else:
            vals = [f[key] for f in features_list]
            batch_dict[key] = torch.as_tensor(np.stack(vals), device=device)

    return batch_dict

def get_step_data(dataset, trajectory_id, base_index):
    data = {}
    dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
    # Get the data for all modalities
    for modality in dataset.modality_keys:
        # Get the data corresponding to each key in the modality
        for key in dataset.modality_keys[modality]:
            if modality == "video":
                pass
            elif modality == "state" or modality == "action":
                data[key] = dataset.get_state_or_action(trajectory_id, modality, key, base_index)
            elif modality == "language":
                data[key] = dataset.get_language(trajectory_id, key, base_index)

    return data

def save_trajectory_data(trajectory_data, dataset, trajectory_id, output_dir):
    chunk_index = dataset.get_episode_chunk(trajectory_id)
    chunk_dir = f"chunk-{chunk_index:03d}"
    os.makedirs(os.path.join(output_dir, "data", chunk_dir), exist_ok=True)

    episode_id = f"episode_{int(trajectory_id):06d}"
    output_file_path = os.path.join(output_dir, "data", chunk_dir, f"{episode_id}.parquet")
    trajectory_data.to_parquet(output_file_path)

##
# Worker function, now handles either validation or writing based on a boolean flag.
##
def worker_func(
    gpu_id: int,
    traj_id_list: list,
    checkpoint_path: str,
    validation_dataset_path: str,
    output_dir: str,
    batch_size: int,
    num_workers: int = 1,
    video_indices=None,
    dataset=None,
    modality_configs=None,
):
    """This function runs in a separate process on GPU `gpu_id` and processes all `traj_id_list`."""
    # Load model on this GPU
    model = IDM.from_pretrained(checkpoint_path)

    model.requires_grad_(False)
    model.eval()
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)

    for tid in tqdm(traj_id_list, desc=f"GPU {gpu_id}", position=gpu_id):
        action_dict = {}
        traj_data = dataset.get_trajectory_data(tid)
        length = len(traj_data)


        all_features = []

        # Create a simple prefetching mechanism with a thread pool
        from concurrent.futures import ThreadPoolExecutor

        def load_and_transform_step(step_idx, video_data):
            step_data = get_step_data(dataset, tid, step_idx)
            timestamp = dataset.curr_traj_data["timestamp"].to_numpy()
            for key in video_data:
                frames, whole_indices = video_data[key]
                step_indices = dataset.delta_indices[key] + step_idx
                step_indices = np.maximum(step_indices, 0)
                step_indices = np.minimum(step_indices, dataset.trajectory_lengths[tid] - 1)
                indices = np.array([np.where(np.isclose(whole_indices, val))[0][0] for val in timestamp[step_indices]])
                step_data[key] = frames[indices]

            output = dataset.transforms(step_data)
            return output


        video_data = {}
        for key in dataset.modality_keys["video"]:
            video_path = dataset.get_video_path(tid, key.replace("video.", ""))
            video_backend = dataset.video_backend
            video_backend_kwargs = dataset.video_backend_kwargs
            frames, whole_indices = get_all_frames_and_timestamps(video_path.as_posix(), video_backend, video_backend_kwargs)
            video_data[key] = (frames, whole_indices)


        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for step_idx in range(length):
                future = executor.submit(load_and_transform_step, step_idx, video_data)
                futures.append(future)

            all_features = []
            for future in as_completed(futures):
                all_features.append(future.result())


        for start_idx in range(0, length, batch_size):
            end_idx = min(start_idx + batch_size, length)
            step_ids = list(range(start_idx, end_idx))

            batch_features = all_features[start_idx:end_idx]
            batch_dict = collate_fn(batch_features, device)

            with torch.no_grad():
                out = model.get_action(batch_dict)

            pred_actions = out["action_pred"].cpu()
            pred_actions = dataset.transforms.unapply(Batch(action=pred_actions))

            # Load modality.json to get the proper structure
            modality_json_path = os.path.join(validation_dataset_path, 'meta', 'modality.json')

            with open(modality_json_path, 'r') as f:
                modality_config = json.load(f)

            # Get action part configurations
            action_parts = modality_config.get('action', {})

            # Calculate total action dimension
            total_dim = 0
            for part, indices in action_parts.items():
                total_dim = max(total_dim, indices.get('end', 0))

            # Check if we have an action horizon dimension
            has_horizon = False
            action_horizon = 1
            sample_key = list(pred_actions.keys())[0]
            if len(pred_actions[sample_key].shape) == 3:
                has_horizon = True
                batch_size, action_horizon, _ = pred_actions[sample_key].shape
            else:
                batch_size = pred_actions[sample_key].shape[0]

            if has_horizon:
                final_actions = np.zeros((batch_size, action_horizon, total_dim))

                # Fill in the actions for parts that exist in pred_actions
                for part, indices in action_parts.items():
                    action_key = f'action.{part}'
                    start_idx = indices.get('start', 0)
                    end_idx = indices.get('end', 0)

                    if action_key in pred_actions:
                        final_actions[:, :, start_idx:end_idx] = pred_actions[action_key]
            else:
                final_actions = np.zeros((batch_size, total_dim))

                # Fill in the actions for parts that exist in pred_actions
                for part, indices in action_parts.items():
                    action_key = f'action.{part}'
                    start_idx = indices.get('start', 0)
                    end_idx = indices.get('end', 0)

                    if action_key in pred_actions:
                        final_actions[:, start_idx:end_idx] = pred_actions[action_key]

            pred_actions = final_actions

            # Not validating => we do the usual writing
            for i, s in enumerate(step_ids):
                for j in range(action_horizon):
                    if s+j >= length:
                        break
                    if s+j not in action_dict:
                        action_dict[s+j] = []
                    if has_horizon:
                        action_dict[s+j].append(pred_actions[i, j].flatten())
                    else:
                        action_dict[s+j].append(pred_actions[i].flatten())

        for s in action_dict:
            mean_action = np.mean(action_dict[s], axis=0)
            traj_data.at[s, "action"] = mean_action

        save_trajectory_data(traj_data, dataset, tid, output_dir)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def validate_checkpoint(
    checkpoint_path,
    validation_dataset_path,
    output_dir=None,
    num_gpus=8,
    batch_size=16,
    max_episodes=None,
    num_workers=1,
    video_indices=None,
):
    device_count = torch.cuda.device_count()
    logger.info(f"Found {device_count} GPUs available.")
    if device_count < num_gpus:
        logger.warn(
            f"WARNING: You requested num_gpus={num_gpus} but only {device_count} GPUs are visible."
        )
        num_gpus = device_count

    # If not validation, create the output directory for saving predicted actions
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Will save predicted actions to: {output_dir}")

    _, dataset, modality_configs = load_dataset_and_config(checkpoint_path, validation_dataset_path, video_indices)

    dataset.transforms.eval()

    trajectory_ids = dataset.trajectory_ids
    if max_episodes is not None and max_episodes > 0:
        trajectory_ids = trajectory_ids[:max_episodes]

    logger.debug(f"Processing {len(trajectory_ids)} trajectories total.")

    # Split trajectories among GPUs
    chunk_size = (len(trajectory_ids) + num_gpus - 1) // num_gpus

    tasks_by_gpu = {}
    for i in range(num_gpus):
        start = i * chunk_size
        end = min(start + chunk_size, len(trajectory_ids))
        if start >= end:
            break
        gpu_traj_list = trajectory_ids[start:end]
        tasks_by_gpu[i] = gpu_traj_list

    try:
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id, gpu_traj_list in tasks_by_gpu.items():
                future = executor.submit(
                    worker_func,
                    gpu_id,
                    gpu_traj_list,
                    checkpoint_path,
                    validation_dataset_path,
                    output_dir,
                    batch_size,
                    num_workers,
                    video_indices,
                    dataset,
                    modality_configs,
                )
                futures.append(future)

            # Wait for all futures to complete or handle interruption
            for future in as_completed(futures):
                try:
                    # Get the result to catch any exceptions
                    future.result()
                except Exception as e:
                    logger.error(f"Error in worker process: {e}")
                    import traceback
                    traceback.logger.debug_exc()
    except KeyboardInterrupt:
        logger.error("\nProcess interrupted by user. Cleaning up...")
        # The context manager will handle cancellation of pending futures
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        import traceback
        traceback.logger.debug_exc()
    finally:
        # Force cleanup of CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # If not validating, optionally copy the meta & videos folders
    if output_dir:
        # Copy the metadata directory
        meta_src = os.path.join(validation_dataset_path, "meta")
        meta_dst = os.path.join(output_dir, "meta")
        if os.path.exists(meta_src):
            import shutil

            if not os.path.exists(meta_dst):
                shutil.copytree(meta_src, meta_dst)
            logger.info(f"Copied metadata to: {meta_dst}")

            tasks_path = os.path.join(meta_dst, "tasks.jsonl")
            if os.path.exists(tasks_path):
                # Read the existing tasks
                tasks = []
                with open(tasks_path, "r") as f:
                    for line in f:
                        tasks.append(json.loads(line))

                # Update tasks with <DREAM> prefix if not already present
                updated_tasks = []
                for task in tasks:
                    if "task" in task and not task["task"].startswith("<DREAM>"):
                        task["task"] = f"<DREAM>{task['task']}"
                    updated_tasks.append(task)

                # Write the updated tasks back to the file
                with open(tasks_path, "w") as f:
                    for task in updated_tasks:
                        f.write(json.dumps(task) + "\n")

                logger.info("Updated tasks.jsonl with <DREAM> prefix")

        # Copy the videos directory if it exists
        videos_src = os.path.join(validation_dataset_path, "videos")
        videos_dst = os.path.join(output_dir, "videos")
        if os.path.exists(videos_src):
            import shutil

            if not os.path.exists(videos_dst):
                shutil.copytree(videos_src, videos_dst)
            logger.info(f"Copied videos to: {videos_dst}")

    return {"message": "Dataset with predicted actions written to disk"}

# raw_to_lerobot
CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

def get_video_metadata(video_path):
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=height,width,codec_name,pix_fmt,r_frame_rate",
        "-of", "json", video_path,
    ]

    try:
        output = subprocess.check_output(cmd).decode("utf-8")
        probe_data = json.loads(output)
        stream = probe_data["streams"][0]

        # Parse frame rate
        num, den = map(int, stream["r_frame_rate"].split("/"))
        fps = num / den

        return {
            "dtype": "video",
            "shape": [stream["height"], stream["width"], 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": fps,
                "video.codec": stream["codec_name"],
                "video.pix_fmt": stream["pix_fmt"],
                "video.is_depth_map": False,
            },
        }
    except Exception as e:
        logger.error(f"Error getting video metadata: {e}")
        return None

def dump_jsonl(data: List[Dict[str, Any]], path: Path) -> None:
    """Dump list of dictionaries as JSONL file."""
    with open(path, 'w') as f:
        for item in data:
            json_str = json.dumps(item)
            f.write(json_str + '\n')

def json_dump(data: Dict[str, Any], path: Path, indent: int = 4) -> None:
    """Dump dictionary as JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def process_video_chunk(args):
    """Process a chunk of videos in parallel."""
    video_files, labels_dir, output_dir, wan, data_type, videos_dir, video_key = args
    results = []

    for video_file in video_files:
        video_id = video_file.stem
        label_file = labels_dir / f"{video_id}.txt"

        # Read annotation
        with open(label_file, "r") as f:
            annotation = f.read().strip()

        # Get video frame count (if not in wan mode)
        if wan:
            frame_count = 81  # Fixed frame count for wan
        else:
            try:
                cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
                      "-show_entries", "stream=duration,r_frame_rate", "-of", "json", str(videos_dir / video_key / f"{video_file}.mp4")]
                output = subprocess.check_output(cmd).decode()
                probe_data = json.loads(output)

                if "streams" in probe_data and len(probe_data["streams"]) > 0:
                    stream = probe_data["streams"][0]

                    # Check if duration is available
                    if "duration" in stream:
                        duration = float(stream["duration"])
                        fps_str = stream.get("r_frame_rate", "30/1")
                        try:
                            num, den = map(int, fps_str.split("/"))
                            fps = num / den
                            frame_count = int(duration * fps)
                        except Exception:
                            # Default to 30fps if fps parsing fails
                            frame_count = int(duration * 30)
                    else:
                        # If no duration, use a default frame count
                        logger.warn(f"Warning: Could not determine duration for {video_file}, using default frame count")
                        frame_count = 81
                else:
                    # No streams found
                    logger.warn(f"Warning: No valid video streams found in {video_file}, using default frame count")
                    frame_count = 81

            except Exception as e:
                logger.error(f"Error getting frame count for {video_file}: {e}")
                # Default frame count if all methods fail
                frame_count = 81

        # Ensure frame count is at least 1
        frame_count = max(1, frame_count)

        results.append((video_id, annotation, frame_count))

    return results


def copy_videos_parallel(video_copy_tasks, max_workers=16):
    """Copy multiple videos in parallel using ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for source, dest in video_copy_tasks:
            futures.append(executor.submit(shutil.copy2, source, dest))

        # Wait for all copy operations to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error copying file: {e}")

def process_folder(args):
    """Process a single input folder."""
    folder_path, output_base_dir, annotation_source, fps, max_videos, num_workers_per_folder, wan, data_type, embodiment, video_key = args

    # Create folder-specific output directory
    folder_name = folder_path.name

    output_dir = output_base_dir / f"{embodiment}.{folder_name}"

    # Process this folder
    result = convert_raw_to_lerobot(
        raw_dir=folder_path,
        output_dir=output_dir,
        annotation_source=annotation_source,
        fps=fps,
        max_videos=max_videos,
        num_workers=num_workers_per_folder,
        wan=wan,
        data_type=data_type,
        video_key=video_key,
    )

    return folder_name, result

def convert_raw_to_lerobot(
    raw_dir: Path,
    output_dir: Path,
    annotation_source: str = "human",
    fps: int = 8,
    max_videos: int | None = None,
    num_workers: int = None,
    wan: bool = False,
    data_type: str = "lapa",
    video_key: str = None,
):
    """Convert raw dataset to LeRobot format."""
    # Setup directories
    videos_dir = raw_dir / "videos"
    labels_dir = raw_dir / "labels"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata directory
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Check for the new folder structure
    video_subfolders = [f for f in videos_dir.iterdir() if f.is_dir()]
    if video_subfolders:
        # New folder structure detected
        logger.debug(f"Detected new folder structure with subfolders: {[f.name for f in video_subfolders]}")

        # Get all unique video IDs across all subfolders
        all_video_ids = set()
        for folder in video_subfolders:
            video_files = folder.glob("*.mp4")
            all_video_ids.update(video_file.stem for video_file in video_files)

        # Convert to sorted list
        video_ids = sorted(list(all_video_ids))

        # Create dummy video_files list with just the IDs
        video_files = [Path(video_id) for video_id in video_ids]
    else:
        raise ValueError("No video subfolders found in the input directory")

    if max_videos is not None:
        video_files = video_files[:max_videos]
        logger.debug(f"Processing only first {max_videos} videos for debugging")

    logger.debug(f"Processing {len(video_files)} videos in {videos_dir}")

    # Setup multiprocessing
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 2)  # Use half of available cores by default

    # Split videos into chunks for parallel processing
    chunk_size = math.ceil(len(video_files) / num_workers)
    video_chunks = [video_files[i:i + chunk_size] for i in range(0, len(video_files), chunk_size)]

    # Prepare arguments for parallel processing
    args_list = [
        (chunk, labels_dir, output_dir, wan, data_type, videos_dir, video_key)
        for chunk in video_chunks
    ]

    # Process videos in parallel
    total_frames = 0
    annotation_to_index = {}
    episodes_info = []

    logger.info(f"Processing {len(video_files)} videos in {raw_dir} using {num_workers} workers")
    logger.info(f"wan mode: {wan} (fixed FPS=8, frames=81)" if wan else "")
    with Pool(num_workers) as pool:
        all_results = list(tqdm(
            pool.imap(process_video_chunk, args_list),
            total=len(args_list),
            desc=f"Processing {raw_dir.name}"
        ))

    # Process results and create videos
    logger.info(f"Processing {len(all_results)} chunks of videos...")

    # We'll collect all video copy tasks here
    all_video_copy_tasks = []


    for chunk_idx, chunk_results in enumerate(all_results):
        for video_id, annotation, frame_count in tqdm(
            chunk_results,
            desc=f"Processing chunk {chunk_idx + 1}/{len(all_results)}",
            leave=False
        ):
            if annotation not in annotation_to_index:
                annotation_to_index[annotation] = len(annotation_to_index)

            episode_index = len(episodes_info)

            # If wan mode is enabled, use fixed frame count
            actual_frame_count = 81 if wan else frame_count
            actual_fps = 8 if wan else fps

            # Create episode data
            episode_data = {
                "observation.state": [np.zeros(44, dtype=np.float32)] * actual_frame_count,
                "action": [np.zeros(44, dtype=np.float32)] * actual_frame_count,
                "timestamp": [i/actual_fps for i in range(actual_frame_count)],
                "episode_index": [episode_index] * actual_frame_count,
                "index": np.arange(total_frames, total_frames + actual_frame_count),
                "task_index": [annotation_to_index[annotation]] * actual_frame_count,
                f"annotation.{annotation_source}": [[annotation_to_index[annotation]]] * actual_frame_count
            }

            # Save episode data
            episode_chunk = episode_index // CHUNKS_SIZE
            data_path = DATA_PATH.format(episode_chunk=episode_chunk, episode_index=episode_index)
            save_path = output_dir / data_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(episode_data)
            df.to_parquet(save_path)


            view_list = []
            for folder in video_subfolders:
                source_video_path = folder / f"{video_id}.mp4"
                if source_video_path.exists():
                    view_list.append((folder.name, source_video_path))

            for video_key, source_video_path in view_list:
                # Get the LeRobot format key

                # Destination path in LeRobot format
                video_save_path = output_dir / VIDEO_PATH.format(
                    episode_chunk=episode_chunk,
                    video_key=video_key,
                    episode_index=episode_index
                )
                video_save_path.parent.mkdir(parents=True, exist_ok=True)

                # Add to copy tasks instead of copying immediately
                all_video_copy_tasks.append((source_video_path, video_save_path))

            # Update episodes info
            episodes_info.append({
                "episode_index": episode_index,
                "tasks": [annotation],
                "length": actual_frame_count,
                "video_id": video_id
            })

            total_frames += actual_frame_count

    # Now copy all videos in parallel
    logger.info(f"Copying {len(all_video_copy_tasks)} videos in parallel...")
    copy_videos_parallel(all_video_copy_tasks, max_workers=min(32, num_workers))

    # Generate metadata files
    # 1. tasks.jsonl
    tasks_path = meta_dir / "tasks.jsonl"
    tasks = [{"task_index": idx, "task": task} for task, idx in annotation_to_index.items()]
    dump_jsonl(tasks, tasks_path)

    # 2. episodes.jsonl
    episodes_path = meta_dir / "episodes.jsonl"
    dump_jsonl(episodes_info, episodes_path)

    # 3. info.json
    # Find a sample video for metadata from each view
    info = {
        "robot_type": data_type,
        "total_episodes": len(video_files),
        "total_frames": total_frames,
        "total_tasks": len(annotation_to_index),
        "total_videos": len(video_subfolders),
        "chunks_size": CHUNKS_SIZE,
        "total_chunks": (len(video_files) + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        "fps": 8 if wan else fps,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": (44,),
                "names": [f"motor_{i}" for i in range(44)]
            },
            "action": {
                "dtype": "float32",
                "shape": (44,),
                "names": [f"motor_{i}" for i in range(44)]
            },
            f"annotation.{annotation_source}": {
                "dtype": "int64",
                "shape": (1,)
            }
        }
    }


    for folder in video_subfolders:
        # Find the first video in this folder
        sample_videos = list(folder.glob("*.mp4"))
        if sample_videos:
            view_key = folder.name
            sample_path = sample_videos[0]
            info["features"][view_key] = get_video_metadata(sample_path)


    # Remove None values from features
    info["features"] = {k: v for k, v in info["features"].items() if v is not None}

    info_path = meta_dir / "info.json"
    json_dump(info, info_path, indent=4)

    return output_dir


# preprocess_video
def custom_crop_pad_resize_gr1(img, target_size=(256, 256)):
    """
    Custom crop, pad, and resize operation that maintains the aspect ratio.

    Args:
        img: Input frame
        original_width: Original video width
        original_height: Original video height
        target_size: Target resolution (width, height)

    Returns:
        Processed frame at target_size resolution
    """
    # For 832x480 videos, adjust the crop parameters proportionally from the 1280x800 example
    # Original crop for 1280x800: (310, 770, 110, 1130) - (top, bottom, left, right)
    original_height, original_width = img.shape[:2]


    # Calculate proportional crop values
    crop_top_ratio = 310 / 800
    crop_bottom_ratio = 770 / 800
    crop_left_ratio = 110 / 1280
    crop_right_ratio = 1130 / 1280

    # Apply ratios to the current dimensions
    crop_top = int(original_height * crop_top_ratio)
    crop_bottom = int(original_height * crop_bottom_ratio)
    crop_left = int(original_width * crop_left_ratio)
    crop_right = int(original_width * crop_right_ratio)

    # Ensure crop boundaries are within image dimensions
    crop_top = max(0, min(crop_top, original_height - 1))
    crop_bottom = max(crop_top + 1, min(crop_bottom, original_height))
    crop_left = max(0, min(crop_left, original_width - 1))
    crop_right = max(crop_left + 1, min(crop_right, original_width))

    # Crop the image
    img_cropped = img[crop_top:crop_bottom, crop_left:crop_right]

    # Calculate intermediate size while maintaining aspect ratio
    cropped_height, cropped_width = img_cropped.shape[:2]
    aspect_ratio = cropped_width / cropped_height

    # Resize to intermediate size (similar to 720x480 in the example)
    intermediate_height = 480
    intermediate_width = 720
    img_resized = cv2.resize(img_cropped, (intermediate_width, intermediate_height), cv2.INTER_AREA)

    # Pad to make it square
    if intermediate_width > intermediate_height:
        # Width is larger, pad height
        height_pad = (intermediate_width - intermediate_height) // 2
        img_pad = np.pad(
            img_resized, ((height_pad, height_pad), (0, 0), (0, 0)), mode="constant", constant_values=0
        )
    else:
        # Height is larger, pad width
        width_pad = (intermediate_height - intermediate_width) // 2
        img_pad = np.pad(
            img_resized, ((0, 0), (width_pad, width_pad), (0, 0)), mode="constant", constant_values=0
        )

    # Final resize to target size
    final_img = cv2.resize(img_pad, target_size, cv2.INTER_AREA)

    return final_img

def resize_with_padding(img, ratio=1.0, target_size=(256, 256)):
    # Original aspect ratio is 1280:800 (or 16:10)
    target_ratio = ratio

    h, w = img.shape[:2]
    current_ratio = w / h
    if target_ratio >= 1:
        # Width is the limiting factor
        new_w = target_size[0]
        new_h = int(new_w / target_ratio)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Add padding to height
        pad_top = (target_size[1] - new_h) // 2
        pad_bottom = target_size[1] - new_h - pad_top
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        # Height is the limiting factor
        new_h = target_size[1]
        new_w = int(new_h * target_ratio)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Add padding to width
        pad_left = (target_size[0] - new_w) // 2
        pad_right = target_size[0] - new_w - pad_left
        padded = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded

def extract_subimages_franka(frame, original_width, original_height):
    h, w = frame.shape[:2]  # h=480, w=832

    # Calculate dimensions for even division
    half_width = w // 2
    half_height = h // 2

    # Extract subimages
    image_side_0 = frame[:half_height, :half_width]     # Top-left (240x416)
    image_side_1 = frame[:half_height, half_width:]     # Top-right (240x416)
    wrist_image = frame[half_height:, :half_width]      # Bottom-left (240x416)


    image_side_0 = cv2.resize(image_side_0, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    image_side_1 = cv2.resize(image_side_1, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    wrist_image = cv2.resize(wrist_image, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

    return image_side_0, image_side_1, wrist_image

def extract_subimages(frame, ratio):
    """Extract subimages from a frame and resize to 256x256 while preserving aspect ratio with padding."""
    h, w = frame.shape[:2]  # h=480, w=832

    # Calculate dimensions for even division
    half_width = w // 2  # 416
    half_height = h // 2  # 240

    # Extract subimages
    image_side_0 = frame[:half_height, :half_width]     # Top-left (240x416)
    image_side_1 = frame[:half_height, half_width:]     # Top-right (240x416)
    wrist_image = frame[half_height:, :half_width]      # Bottom-left (240x416)

    image_side_0 = resize_with_padding(image_side_0, ratio)
    image_side_1 = resize_with_padding(image_side_1, ratio)
    wrist_image = resize_with_padding(wrist_image, ratio)

    return image_side_0, image_side_1, wrist_image

def process_batch_frames(frames, output_videos, src_path, dataset, original_width, original_height):
    """Process a batch of frames."""
    ratio = original_width / original_height
    for frame in frames:
        # Extract subimages
        if dataset == 'robocasa':
            image_side_0, image_side_1, wrist_image = extract_subimages(frame, ratio)
            output_videos['observation.images.left_view'].write(image_side_0)
            output_videos['observation.images.right_view'].write(image_side_1)
            output_videos['observation.images.wrist_view'].write(wrist_image)
        elif dataset == 'gr1':
            image = custom_crop_pad_resize_gr1(frame)
            output_videos['observation.images.ego_view'].write(image)
        elif dataset == 'franka':
            image_side_0, image_side_1, wrist_image = extract_subimages(frame, ratio)
            output_videos['observation.images.exterior_image_1_left_pad_res256_freq15'].write(image_side_0)
            output_videos['observation.images.exterior_image_2_left_pad_res256_freq15'].write(image_side_1)
            output_videos['observation.images.wrist_image_left_pad_res256_freq15'].write(wrist_image)
        elif dataset == 'so100':
            image = resize_with_padding(frame, ratio)
            output_videos['observation.images.webcam'].write(image)
        else:
            raise ValueError(f"Unknown task: {src_path}")

def process_video(args):
    """Process a single video file."""
    src_path, dst_dir, video_name, dataset, original_width, original_height = args

    # Create output directories if they don't exist
    if dataset == 'robocasa':
        output_dirs = {
            'observation.images.left_view': os.path.join(dst_dir, 'videos', 'observation.images.left_view'),
            'observation.images.right_view': os.path.join(dst_dir, 'videos', 'observation.images.right_view'),
            'observation.images.wrist_view': os.path.join(dst_dir, 'videos', 'observation.images.wrist_view'),
        }
    elif dataset == 'gr1':
        output_dirs = {
            'observation.images.ego_view': os.path.join(dst_dir, 'videos', 'observation.images.ego_view')
        }
    elif dataset == 'franka':
        output_dirs = {
            'observation.images.exterior_image_1_left_pad_res256_freq15': os.path.join(dst_dir, 'videos', 'observation.images.exterior_image_1_left_pad_res256_freq15'),
            'observation.images.exterior_image_2_left_pad_res256_freq15': os.path.join(dst_dir, 'videos', 'observation.images.exterior_image_2_left_pad_res256_freq15'),
            'observation.images.wrist_image_left_pad_res256_freq15': os.path.join(dst_dir, 'videos', 'observation.images.wrist_image_left_pad_res256_freq15'),
        }
    elif dataset == 'so100':
        output_dirs = {
            'observation.images.webcam': os.path.join(dst_dir, 'videos', 'observation.images.webcam'),
        }

    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Open the source video
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {src_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create VideoWriter objects for each subimage
    output_videos = {}
    for name, dir_path in output_dirs.items():
        output_path = os.path.join(dir_path, f"{video_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')
        output_videos[name] = cv2.VideoWriter(output_path, fourcc, fps, (256, 256))  # Updated size to 256x256


    # Process frames in batches
    batch_size = 32  # Adjust based on available memory
    frames_batch = []

    # Process each frame with progress bar
    pbar = tqdm(total=frame_count, desc=f"Processing {video_name}", leave=False)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_batch.append(frame)

        # Process batch when it reaches the desired size
        if len(frames_batch) >= batch_size:
            process_batch_frames(frames_batch, output_videos, src_path, dataset, original_width, original_height)
            frames_batch = []
            pbar.update(batch_size)

    # Process remaining frames
    if frames_batch:
        process_batch_frames(frames_batch, output_videos, src_path, dataset, original_width, original_height)
        pbar.update(len(frames_batch))

    # Close progress bar
    pbar.close()

    # Release resources
    cap.release()
    for writer in output_videos.values():
        writer.release()

def copy_labels(src_dir, dst_dir):
    """Copy label files from source to destination."""
    src_labels_dir = os.path.join(src_dir, 'labels')
    dst_labels_dir = os.path.join(dst_dir, 'labels')

    if os.path.exists(src_labels_dir):
        os.makedirs(dst_labels_dir, exist_ok=True)
        for label_file in os.listdir(src_labels_dir):
            if label_file.endswith('.txt'):
                shutil.copy2(
                    os.path.join(src_labels_dir, label_file),
                    os.path.join(dst_labels_dir, label_file)
                )

def process_subdirectory(subdir, src_dir, dst_dir, num_workers, max_videos=None, dataset=None, original_width=None, original_height=None):
    """Process a single subdirectory."""
    logger.debug(f"Processing subdirectory: {src_dir}, {subdir}")
    src_subdir = os.path.join(src_dir, subdir)
    dst_subdir = os.path.join(dst_dir, subdir)

    # Copy label files
    copy_labels(src_subdir, dst_subdir)

    # Process videos
    src_videos_dir = os.path.join(src_subdir, 'videos')
    if os.path.exists(src_videos_dir):
        video_files = [f for f in os.listdir(src_videos_dir) if f.endswith('.mp4')]

        # Limit number of videos if max_videos is specified
        if max_videos is not None:
            video_files = video_files[:max_videos]
            logger.debug(f"Processing limited set of {len(video_files)} videos in {subdir}")

        # Prepare arguments for multiprocessing
        args_list = [
            (os.path.join(src_videos_dir, video_file), dst_subdir, os.path.splitext(video_file)[0], dataset, original_width, original_height)
            for video_file in video_files
        ]

        # Process videos in parallel
        with mp.Pool(num_workers) as pool:
            list(tqdm(pool.imap(process_video, args_list), total=len(args_list), desc=f"Processing {subdir}"))

def process_directory(src_dir, dst_dir, num_workers=None, num_subdirs_parallel=1, max_videos=None, dataset=None, original_width=None, original_height=None, recursive=False):
    """Process all videos in the source directory."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 2)  # Leave some cores for system

    # Create destination directory structure
    os.makedirs(dst_dir, exist_ok=True)


    if recursive:
        # Get all subdirectories in the source directory
        subdirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    else:
        # Get all subdirectories in the source directory
        subdirs = ['']

    # Process subdirectories in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_subdirs_parallel) as executor:
        process_subdir_fn = partial(process_subdirectory,
                                  src_dir=src_dir,
                                  dst_dir=dst_dir,
                                  num_workers=num_workers // num_subdirs_parallel,
                                  max_videos=max_videos,
                                  dataset=dataset,
                                  original_width=original_width,
                                  original_height=original_height)
        list(tqdm(executor.map(process_subdir_fn, subdirs), total=len(subdirs), desc="Processing subdirectories"))

# split_video_instruction
def process_mp4_files(source_dir, output_dir, recursive=False):
    """
    Process MP4 files from source_dir and its subdirectories, extract instructions from filenames,
    and save them to output_dir with appropriate structure.

    Args:
        source_dir: Directory containing MP4 files or subdirectories with MP4 files
        output_dir: Directory to save processed data
        recursive: If True, maintain the directory structure from source_dir in output_dir
    """
    if not recursive:
        # Original behavior: process all MP4 files into a single output directory
        labels_dir = os.path.join(output_dir, "labels")
        videos_dir = os.path.join(output_dir, "videos")

        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(videos_dir, exist_ok=True)

        # Find all MP4 files in source_dir and its subdirectories
        mp4_files = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.mp4'):
                    # Store full path to the file
                    mp4_files.append(os.path.join(root, file))

        # Sort files to ensure consistent ordering
        mp4_files.sort()

        instruction = source_dir.split("/")[-1].replace("_", " ")
        # Process each MP4 file
        for idx, mp4_path in enumerate(mp4_files, 1):
            mp4_file = os.path.basename(mp4_path)


            label_file = os.path.join(labels_dir, f"{idx}.txt")
            with open(label_file, 'w') as f:
                f.write(instruction)

            # Copy video with new name
            target_video = os.path.join(videos_dir, f"{idx}.mp4")
            shutil.copy2(mp4_path, target_video)

            logger.debug(f"Processed {mp4_path} -> {idx}.mp4, instruction: {instruction}")

        logger.debug(f"Processed {len(mp4_files)} files. Results saved to {output_dir}")
    else:
        # Recursive behavior: maintain directory structure
        logger.debug(f"Processing in recursive mode, maintaining directory structure...")

        # Get all subdirectories in the source directory
        subdirs = []
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            if os.path.isdir(item_path):
                subdirs.append(item)

        if not subdirs:
            # If no subdirectories, process the source directory directly
            logger.info(f"No subdirectories found in {source_dir}, processing directly...")
            process_mp4_files(source_dir, output_dir, recursive=False)
            return

        # Process each subdirectory
        for subdir in subdirs:
            source_subdir = os.path.join(source_dir, subdir)
            output_subdir = os.path.join(output_dir, subdir)

            # Create output subdirectory
            os.makedirs(output_subdir, exist_ok=True)

            # Process files in this subdirectory
            process_mp4_files(source_subdir, output_subdir, recursive=False)

        logger.debug(f"Recursive processing complete. Results saved to {output_dir}")

class ModelManagerForResume(ModelManager):
    def load_lora(self, file_path="", state_dict={}, lora_alpha=1.0):
        if isinstance(file_path, list):
            for file_path_ in file_path:
                self.load_lora(file_path_, state_dict=state_dict, lora_alpha=lora_alpha)
        else:
            logger.info(f"Loading LoRA models from file: {file_path}")
            if len(state_dict) == 0:
                state_dict = load_state_dict(file_path)
            new_state_dict = {}
            for name, param in state_dict["state_dict"].items():
                name = name.replace("pipe.dit.", "")
                new_state_dict[name] = param
            state_dict = new_state_dict
            for model_name, model, model_path in zip(self.model_name, self.model, self.model_path):
                for lora in get_lora_loaders():
                    match_results = lora.match(model, state_dict)
                    if match_results is not None:
                        logger.info(f"    Adding LoRA to {model_name} ({model_path}).")
                        lora_prefix, model_resource = match_results
                        lora.load(model, state_dict, lora_prefix, alpha=lora_alpha, model_resource=model_resource)
                        break

def main(task_id, save_path, dataset, embodiment, annotation_source, global_metadata_path, fps, is_wan, lerobot_data_type, idm_checkpoint_path):
    # step 1: split_video_instruction
    split_video_instruction_src_path = os.path.dirname(save_path)
    split_video_instruction_dst_path = split_video_instruction_src_path + f"/{task_id}/{dataset}_data"
    # step 2: preprocess_video
    preprocess_video_src_dir = split_video_instruction_dst_path
    preprocess_video_dst_dir = split_video_instruction_src_path + f"/{task_id}/{dataset}_data_split"
    original_width = 832 
    original_height = 480
    workers = 8
    parallel_subdirs = 1
    max_videos = None
    # step 3: raw_to_lerobot
    raw_to_lerobot_src_dir = preprocess_video_dst_dir
    raw_to_lerobot_dst_dir = split_video_instruction_src_path + f"/{task_id}/{embodiment}.data"
    video_fps = fps
    video_key = None
    # step 4: dump_idm_actions
    dump_idm_actions_src_dir = raw_to_lerobot_dst_dir
    dump_idm_actions_dst_dir = split_video_instruction_src_path + f"/{task_id}/{embodiment}.data_idm"
    max_episodes = None
    num_gpus = 1
    batch_size = 32
    video_indices = "0 8"
    # step 5: compress lerobot dataset
    dump_idm_actions_zip_path = dump_idm_actions_dst_dir

    logger.info(f'begin split_video_instruction')
    process_mp4_files(split_video_instruction_src_path, split_video_instruction_dst_path)

    logger.info(f'begin preprocess_video')
    # Set OpenCV thread optimization
    cv2.setNumThreads(1)
    process_directory(preprocess_video_src_dir, preprocess_video_dst_dir, workers, parallel_subdirs, max_videos=max_videos,
                      dataset=dataset, original_width=original_width, original_height=original_height, recursive=False)

    logger.info(f'begin raw_to_lerobot')
    convert_raw_to_lerobot(
        raw_dir=Path(raw_to_lerobot_src_dir),
        output_dir=Path(raw_to_lerobot_dst_dir),
        annotation_source=annotation_source,
        fps=video_fps,
        max_videos=max_videos,
        num_workers=workers,
        wan=is_wan,
        data_type=lerobot_data_type,
        video_key=video_key
    )
    shutil.copy(global_metadata_path + "/modality.json", raw_to_lerobot_dst_dir + "/meta/modality.json")
    shutil.copy(global_metadata_path + "/stats.json", raw_to_lerobot_dst_dir + "/meta/stats.json")

    logger.info(f'begin dump_idm_action')
    mp.set_start_method("spawn", force=True)
    validate_checkpoint(
        checkpoint_path=idm_checkpoint_path,
        validation_dataset_path=dump_idm_actions_src_dir,
        output_dir=dump_idm_actions_dst_dir,
        num_gpus=num_gpus,
        batch_size=batch_size,
        max_episodes=max_episodes,
        num_workers=1 if num_gpus == 1 else workers,
        video_indices=video_indices,
    )

    logger.info(f'begin compress_directory_to_zip')
    compress_directory_to_zip(dump_idm_actions_dst_dir, dump_idm_actions_zip_path)
    logger.info(f'Saved lerobot dataset to {dump_idm_actions_zip_path}')


if __name__ == "__main__":
    fire.Fire(main)