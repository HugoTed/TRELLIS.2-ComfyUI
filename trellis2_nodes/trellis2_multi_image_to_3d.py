from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from trellis2_client.worker_client import ensure_worker_running, generate_glb

from .trellis2_image_to_3d import ComfyTypes, _file3d_from_path, _pil_to_b64, _to_model_file


def _batch_to_pils(image) -> list[Image.Image]:
    """Convert a ComfyUI IMAGE tensor [B,H,W,C] into a list of PIL images."""
    pils = []
    for i in range(image.shape[0]):
        arr = image[i].detach().cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 4:
            pils.append(Image.fromarray(arr, mode="RGBA"))
        else:
            pils.append(Image.fromarray(arr, mode="RGB").convert("RGBA"))
    return pils


class Trellis2MultiImageTo3D:
    """Generate a textured GLB from multiple views of the same object using TRELLIS.2.

    Uses the official tuning-free multi-image algorithm (stochastic / multidiffusion
    condition fusion, same as TRELLIS `run_multi_image`).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "multi_image_mode": (["stochastic", "multidiffusion"], {"default": "multidiffusion"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "resolution": (["512", "1024", "1536"], {"default": "512"}),
                "max_num_tokens": ("INT", {"default": 16384, "min": 4096, "max": 49152, "step": 4096}),
                "ss_steps": ("INT", {"default": 8, "min": 1, "max": 50}),
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "shape_steps": ("INT", {"default": 8, "min": 1, "max": 50}),
                "shape_guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "tex_steps": ("INT", {"default": 8, "min": 1, "max": 50}),
                "decimation_target": ("INT", {"default": 200000, "min": 1000, "max": 2000000, "step": 1000}),
                "texture_size": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 256}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "preprocess_image": ("BOOLEAN", {"default": True}),
                "remesh": ("BOOLEAN", {"default": False}),
                "bake_normal_map": ("BOOLEAN", {"default": True}),
                "bake_ao": ("BOOLEAN", {"default": True}),
                "ao_samples": ("INT", {"default": 32, "min": 4, "max": 256, "step": 4}),
                "texture_format": (["png", "webp"], {"default": "png"}),
                "setup_status": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FILE_3D_GLB") if ComfyTypes else ("STRING", "STRING")
    RETURN_NAMES = ("model_file", "status", "model_3d") if ComfyTypes else ("model_file", "status")
    FUNCTION = "generate"
    CATEGORY = "3d/trellis2"
    OUTPUT_NODE = True

    def generate(
        self,
        images,
        multi_image_mode: str,
        seed: int,
        resolution: str,
        max_num_tokens: int,
        ss_steps: int,
        ss_guidance_strength: float,
        shape_steps: int,
        shape_guidance_strength: float,
        tex_steps: int,
        decimation_target: int,
        texture_size: int,
        image_2=None,
        image_3=None,
        image_4=None,
        preprocess_image: bool = True,
        remesh: bool = False,
        bake_normal_map: bool = True,
        bake_ao: bool = True,
        ao_samples: int = 32,
        texture_format: str = "png",
        setup_status: str = "",
    ):
        pipeline_type = {
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
        }[resolution]

        pils = _batch_to_pils(images)
        for extra in (image_2, image_3, image_4):
            if extra is not None:
                pils.extend(_batch_to_pils(extra))

        if len(pils) < 2:
            raise ValueError(
                "Trellis2MultiImageTo3D needs at least 2 views. "
                "Batch multiple images into `images` (e.g. via Image Batch) or connect image_2/3/4. "
                "For a single image use the TRELLIS.2 Image to 3D node."
            )

        ensure_worker_running()

        payload = {
            "images_b64": [_pil_to_b64(p) for p in pils],
            "multi_image_mode": multi_image_mode,
            "seed": seed,
            "pipeline_type": pipeline_type,
            "max_num_tokens": max_num_tokens,
            "preprocess_image": preprocess_image,
            "ss_steps": ss_steps,
            "ss_guidance_strength": ss_guidance_strength,
            "shape_steps": shape_steps,
            "shape_guidance_strength": shape_guidance_strength,
            "tex_steps": tex_steps,
            "decimation_target": decimation_target,
            "texture_size": texture_size,
            "remesh": remesh,
            "bake_normal_map": bake_normal_map,
            "bake_ao": bake_ao,
            "ao_samples": ao_samples,
            "texture_format": texture_format,
        }
        result = generate_glb(payload)
        model_file = _to_model_file(result)
        filepath = result.get("filepath", model_file)
        status = f"GLB saved ({len(pils)} views, {multi_image_mode}): {filepath}"
        if setup_status:
            status = f"{setup_status}\n{status}"

        if ComfyTypes is not None:
            model_3d = _file3d_from_path(str(Path(filepath).resolve()))
            return (model_file, status, model_3d)

        return {
            "ui": {"result": [model_file, None, None]},
            "result": (model_file, status),
        }
