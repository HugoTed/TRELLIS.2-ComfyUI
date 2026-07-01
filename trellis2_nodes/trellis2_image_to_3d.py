from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image

from trellis2_client.worker_client import ensure_worker_running, generate_glb

try:
    from comfy_api.latest import Types as ComfyTypes
except ImportError:
    ComfyTypes = None


def _tensor_to_pil(image) -> Image.Image:
    arr = image[0].detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        return Image.fromarray(arr, mode="RGBA")
    return Image.fromarray(arr, mode="RGB").convert("RGBA")


def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _to_model_file(result: dict) -> str:
    filename = result["filename"]
    filepath = result.get("filepath", filename)
    try:
        import folder_paths

        output_dir = os.path.normpath(folder_paths.get_output_directory())
        full = os.path.normpath(filepath)
        if full.startswith(output_dir):
            return os.path.relpath(full, output_dir).replace("\\", "/")
    except ImportError:
        pass
    return filepath.replace("\\", "/")


def _file3d_from_path(path: str):
    if ComfyTypes is None:
        return path
    return ComfyTypes.File3D(path, file_format="glb")


class Trellis2ImageTo3D:
    """Generate a textured GLB from a single image using TRELLIS.2."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "resolution": (["512", "1024", "1536"], {"default": "1024"}),
                "ss_steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "shape_steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "shape_guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "tex_steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "decimation_target": ("INT", {"default": 500000, "min": 1000, "max": 2000000, "step": 1000}),
                "texture_size": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
            },
            "optional": {
                "preprocess_image": ("BOOLEAN", {"default": True}),
                "remesh": ("BOOLEAN", {"default": True}),
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
        image,
        seed: int,
        resolution: str,
        ss_steps: int,
        ss_guidance_strength: float,
        shape_steps: int,
        shape_guidance_strength: float,
        tex_steps: int,
        decimation_target: int,
        texture_size: int,
        preprocess_image: bool = True,
        remesh: bool = True,
        setup_status: str = "",
    ):
        pipeline_type = {
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
        }[resolution]

        ensure_worker_running()

        pil = _tensor_to_pil(image)
        payload = {
            "image_b64": _pil_to_b64(pil),
            "seed": seed,
            "pipeline_type": pipeline_type,
            "preprocess_image": preprocess_image,
            "ss_steps": ss_steps,
            "ss_guidance_strength": ss_guidance_strength,
            "shape_steps": shape_steps,
            "shape_guidance_strength": shape_guidance_strength,
            "tex_steps": tex_steps,
            "decimation_target": decimation_target,
            "texture_size": texture_size,
            "remesh": remesh,
        }
        result = generate_glb(payload)
        model_file = _to_model_file(result)
        filepath = result.get("filepath", model_file)
        status = f"GLB saved: {filepath}"
        if setup_status:
            status = f"{setup_status}\n{status}"

        if ComfyTypes is not None:
            model_3d = _file3d_from_path(str(Path(filepath).resolve()))
            return (model_file, status, model_3d)

        return {
            "ui": {"result": [model_file, None, None]},
            "result": (model_file, status),
        }
