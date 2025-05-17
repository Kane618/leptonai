import os
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import math
import uuid
import yaml
import shutil
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from multiprocessing import Pool
import subprocess
from tqdm import tqdm

from loguru import logger
import leptonai
from leptonai import ObjectStore
from leptonai.photon import Photon, Worker, File
from leptonai.util import is_valid_url


class LeRobot(Worker):

    image = "vlnkane/isaac-gr00t:latest"

    queue_name = "image2video-idm"
    kv_name = "image2video-idm"

    extra_files = {
        "global_metadata.zip" : "global_metadata.zip",
        "video_to_lerobot_dataset.py" : "video_to_lerobot_dataset.py",
    }

    requirement_dependency = [
        "git+https://github.com/Kane618/Neural-Trajectories-IDM.git", # Install groot
        "diffsynth==1.1.7",
        "opencv-python",
        "fire",
    ]

    system_dependency = ["zip"]

    deployment_template = {
        "resource_shape": "gpu.h100-sxm",
        "env": {
            "WAN_MODEL": "Wan-AI/Wan2.1-I2V-14B-480P",
            "LORA_MODEL": "",
            "IDM_LABELING_MODEL": "seonghyeonye/IDM_gr1",
            "OBJECTSTORE_BUCKET": "public",
        },
    }

    OBJECTSTORE_INPUT_PREFIX = "image2video-idm/input"
    OBJECTSTORE_OUTPUT_PREFIX = "image2video-idm/output"

    def init(self):
        from diffsynth import ModelManager, WanVideoPipeline, load_state_dict, get_lora_loaders
        from huggingface_hub import snapshot_download
        import torch
        LEPTON_WORKSPACE_ID = os.environ.get("LEPTON_WORKSPACE_ID", None)
        LEPTON_WORKSPACE_TOKEN = os.environ.get("LEPTON_WORKSPACE_TOKEN", None)
        leptonai.api.v0.workspace.login(LEPTON_WORKSPACE_ID, LEPTON_WORKSPACE_TOKEN)
        logger.info(f'Login into {LEPTON_WORKSPACE_ID} successfully')

        # unzip global_metadata.zip
        result = subprocess.run(["unzip", "./global_metadata.zip", "-d", "./global_metadata"], capture_output=True, text=True)
        logger.info(result.stdout)
        if result.returncode != 0:
            logger.error(f"Error occurred: {result.stderr}")

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
                                logger.info(f"Adding LoRA to {model_name} ({model_path}).")
                                lora_prefix, model_resource = match_results
                                lora.load(model, state_dict, lora_prefix, alpha=lora_alpha, model_resource=model_resource)
                                break

        # download base model
        model = os.environ["WAN_MODEL"]
        local_dir = os.path.expanduser("~/models/base_model")
        logger.info(f"Using model {model}")
        snapshot_download(model, local_dir=local_dir)
        # download lora model
        lora_model = os.environ.get("LORA_MODEL", None)
        logger.info(f"Using lora model {lora_model}")
        if lora_model is not None:
            self.lora_ckpt_directory = os.path.expanduser(f'~/models/lora.ckpt')
            subprocess.run(['wget', lora_model, '-O', self.lora_ckpt_directory])
        # download idm labeling model
        idm_labeling_model = os.environ["IDM_LABELING_MODEL"]
        logger.info(f"Using idm labeling model {idm_labeling_model}")
        suffix = idm_labeling_model.split("/")[-1]
        self.idm_labeling_local_dir = os.path.expanduser(f"~/models/{suffix}")
        os.makedirs(os.path.dirname(self.idm_labeling_local_dir), exist_ok=True)
        snapshot_download(idm_labeling_model, local_dir=self.idm_labeling_local_dir)

        # get the object store
        bucket = os.environ["OBJECTSTORE_BUCKET"]
        logger.info(f"Using bucket {bucket}")
        self._bucket = bucket
        self._object_store = ObjectStore(bucket)

        self.is_wan = True
        self.lerobot_data_type = "dream"
        self.dataset = None
        self.embodiment = None
        self.annotation_source = None
        if "gr1" in suffix:
            self.dataset = "gr1"
            self.embodiment = "gr1_unified"
            self.annotation_source = "human.coarse_action"
        elif "robocasa" in suffix:
            self.dataset = "robocasa"
            self.embodiment = "robocasa_panda_omron"
            self.annotation_source = "human.action.task_description"
        elif "franka" in suffix:
            self.dataset = "franka"
            self.embodiment = "franka"
            self.annotation_source = "language.language_instruction"
        elif "so100" in suffix:
            self.dataset = "so100"
            self.embodiment = "so100"
            self.annotation_source = "human.task_description"
        else:
            raise ValueError(f"Invalid idm labeling model: {self.idm_labeling_local_dir}, dataset must be gr1, robocasa, franka, or so100")
        self.global_metadata_path = os.path.expanduser(f"./global_metadata/{self.dataset}")

        # Load base model and lora model
        model_manager = ModelManagerForResume(device="cuda")
        model_manager.load_models(
            [
                [
                    f"{local_dir}/diffusion_pytorch_model-00001-of-00007.safetensors",
                    f"{local_dir}/diffusion_pytorch_model-00002-of-00007.safetensors",
                    f"{local_dir}/diffusion_pytorch_model-00003-of-00007.safetensors",
                    f"{local_dir}/diffusion_pytorch_model-00004-of-00007.safetensors",
                    f"{local_dir}/diffusion_pytorch_model-00005-of-00007.safetensors",
                    f"{local_dir}/diffusion_pytorch_model-00006-of-00007.safetensors",
                    f"{local_dir}/diffusion_pytorch_model-00007-of-00007.safetensors",
                ],
                f"{local_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                f"{local_dir}/models_t5_umt5-xxl-enc-bf16.pth",
                f"{local_dir}/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch.bfloat16
        )
        if lora_model is not None:
            model_manager.load_lora(self.lora_ckpt_directory, lora_alpha=1.0)

        self.pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
        self.pipe.enable_vram_management(num_persistent_param_in_dit=None) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.
        super().init()

    @staticmethod
    def is_public_bucket(bucket: str) -> bool:
        return bucket == "public"

    def on_task(
        self,
        task_id: str,
        image: str,
        prompt: str,
        seed: Optional[int] = None,
        fps: int = 8,
    ):
        import torch
        import subprocess
        from diffsynth import save_video
        from diffusers.utils import load_image

        if isinstance(image, tuple):
            bucket, key = image
            image = ObjectStore(bucket).get(key, return_url=True)

        image = load_image(image)
        # image2video
        video = self.pipe(
            prompt=prompt,
            negative_prompt='Vibrant colors, overexposed, static, blurry details, text, subtitles, style, artwork, painting, image, still, grayscale, dull, worst quality, low quality, JPEG artifacts, ugly, mutilated, extra fingers, bad hands, bad face, deformed, disfigured, mutated limbs, fused fingers, stagnant image, cluttered background, three legs, many people in the background, walking backwards.',
            input_image=image,
            num_inference_steps=50,
            seed=seed, tiled=True, cfg_scale=5,
            num_frames=81,
        )
        save_path = os.path.expanduser(f"~/output/{task_id}.mp4")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_video(video, save_path, fps=fps, quality=5)
        logger.info(f'Saved video to {save_path}')
        key = f"{self.OBJECTSTORE_OUTPUT_PREFIX}/{task_id}.mp4"
        try:
            output_url = self._object_store.put(key, save_path)
        except Exception as e:
            logger.error(f"Failed to upload output video to {self._bucket}/{key}: {e}")
            raise
        else:
            logger.info(f"Uploaded output video to {self._bucket}/{key}")


        logger.info(f'begin video to lerobot dataset')
        command = [
            "python", "./video_to_lerobot_dataset.py",
            "--task_id", task_id,
            "--save_path", save_path,
            "--dataset", self.dataset,
            "--embodiment", self.embodiment,
            "--annotation_source", self.annotation_source,
            "--global_metadata_path", self.global_metadata_path,
            "--fps", str(fps),
            "--is_wan", str(self.is_wan),
            "--lerobot_data_type", self.lerobot_data_type,
            "--idm_checkpoint_path", self.idm_labeling_local_dir
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        logger.info(result.stdout)
        if result.returncode != 0:
            logger.error(result.stderr)

        dump_idm_actions_zip_path = os.path.dirname(save_path) +  f"/{task_id}/{self.embodiment}.data_idm"
        idm_actions_zip_path = dump_idm_actions_zip_path + ".zip"
        dataset_zip_key = f"{self.OBJECTSTORE_OUTPUT_PREFIX}/{task_id}_dataset.zip"
        try:
            dataset_output_url = self._object_store.put(dataset_zip_key, idm_actions_zip_path)
        except Exception as e:
            logger.error(f"Failed to upload output dataset to {self._bucket}/{dataset_zip_key}: {e}")
            raise
        else:
            logger.info(f"Uploaded output dataset to {self._bucket}/{dataset_zip_key}")

        # remove the output_dir from the server
        output_dir = os.path.dirname(save_path)
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

        if self._object_store.is_public:
            return {"url": output_url, "dataset_url": dataset_output_url}
        else:
            return {"bucket": self._bucket, "key": key, "dataset_key": dataset_zip_key}

    @Photon.handler
    def run(
        self,
        image: str,
        prompt: str,
        seed: Optional[int] = None,
        fps: int = 8,
    ):
        """
        Submits a I2D generation task to the worker. The task will be executed asynchronously.
        """
        if not is_valid_url(image):
            # When the image is not a valid url, upload the image to the object store.
            # so that the worker can download it.
            logger.info(f"image is not a url, uploading to {self._bucket}")

            # Load the conditioning image
            image = File(image)
            key = f"{self.OBJECTSTORE_INPUT_PREFIX}/{uuid.uuid4()}"
            try:
                output_url = self._object_store.put(key, image)
            except Exception as e:
                logger.error(f"Failed to upload input to {self._bucket}/{key}: {e}")
                raise
            else:
                logger.info(f"Uploaded input to {self._bucket}/{key}")

            if self._object_store.is_public:
                image = output_url
            else:
                image = {"bucket": self._bucket, "key": key}

        return self.task_post({
            "image": image,
            "prompt": prompt,
            "seed": seed,
            "fps": fps,
        })

if __name__ == "__main__":
    LeRobot().launch()