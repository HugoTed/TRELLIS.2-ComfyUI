"""TRELLIS.2 inference helpers — runs inside the isolated worker Python env."""

from __future__ import annotations

import base64
import io
import os
import uuid
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def _is_wsl() -> bool:
    try:
        with open("/proc/version", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


# The worker inherits ComfyUI's environment, including its
# PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync — which is broken on WSL2
# (spurious "Allocation on device" with plenty of free VRAM). Likewise,
# expandable_segments relies on CUDA VMM APIs that WSL2 does not support.
# Always choose the allocator ourselves instead of inheriting.
if _is_wsl():
    # max_split_size_mb limits block splitting so large cached blocks stay
    # reusable — important on WSL where reserve growth is capped by
    # Windows-side commit charge and expandable_segments is unsupported.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native,max_split_size_mb:512"
else:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native,expandable_segments:True"

import torch
from PIL import Image

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline

_PIPELINE: Trellis2ImageTo3DPipeline | None = None
_MODEL_ID: str | None = None


def _nvidia_smi_mem() -> str:
    """Global VRAM usage from the driver. On WSL, torch's mem_get_info can
    miss Windows-side usage entirely; nvidia-smi sees the whole GPU."""
    import shutil
    import subprocess

    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return f"nvidia-smi (whole GPU incl. Windows apps): {out.stdout.strip()}"
    except Exception:
        pass
    return "nvidia-smi unavailable"


def _cuda_mem_info() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return (
        f"GPU VRAM (as seen by this process): {free / 1e9:.1f}GB free / {total / 1e9:.1f}GB total "
        f"(allocated {alloc / 1e9:.1f}GB, reserved {reserved / 1e9:.1f}GB). "
        f"{_nvidia_smi_mem()}"
    )


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "allocation on device" in msg


def _raise_cuda_oom(exc: BaseException, stage: str) -> None:
    hint = (
        f"CUDA OOM during {stage}. {_cuda_mem_info()}. "
        "This is GPU VRAM, not system RAM — increasing WSL memory does not help. "
        "Try resolution=512, max_num_tokens=8192, texture_size=1024, remesh=off, fewer steps; "
        "close Windows GPU apps; restart ComfyUI to free VRAM; avoid Trellis2 Load Model before generate."
    )
    raise RuntimeError(hint) from exc


def get_pipeline(model_id: str) -> Trellis2ImageTo3DPipeline:
    global _PIPELINE, _MODEL_ID
    if _PIPELINE is None or _MODEL_ID != model_id:
        _PIPELINE = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
        _PIPELINE.low_vram = True
        _PIPELINE.cuda()
        _MODEL_ID = model_id
    return _PIPELINE


def image_from_base64(data: str) -> Image.Image:
    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def mesh_to_glb(
    mesh,
    *,
    resolution: int,
    decimation_target: int,
    texture_size: int,
    remesh: bool,
) -> bytes:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mesh.simplify(16777216)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=False,
    )
    buf = io.BytesIO()
    glb.export(buf, extension_webp=True, file_type="glb")
    return buf.getvalue()


def generate_glb(payload: dict[str, Any], output_dir: str) -> dict[str, Any]:
    params_log = {k: v for k, v in payload.items() if k not in ("image_b64", "images_b64")}
    if "images_b64" in payload:
        params_log["num_images"] = len(payload["images_b64"])
    print(f"[trellis2-worker] /generate params: {params_log}", flush=True)

    model_id = payload.get("model_id", "microsoft/TRELLIS.2-4B")
    pipeline = get_pipeline(model_id)

    if "images_b64" in payload:
        images = [image_from_base64(data) for data in payload["images_b64"]]
    else:
        images = [image_from_base64(payload["image_b64"])]
    multi_image_mode = payload.get("multi_image_mode", "stochastic")
    seed = int(payload.get("seed", 0))
    pipeline_type = payload.get("pipeline_type", "1024_cascade")
    preprocess_image = bool(payload.get("preprocess_image", True))

    sparse_params = {
        "steps": int(payload.get("ss_steps", 12)),
        "guidance_strength": float(payload.get("ss_guidance_strength", 7.5)),
        "guidance_rescale": float(payload.get("ss_guidance_rescale", 0.7)),
        "rescale_t": float(payload.get("ss_rescale_t", 5.0)),
    }
    shape_params = {
        "steps": int(payload.get("shape_steps", 12)),
        "guidance_strength": float(payload.get("shape_guidance_strength", 7.5)),
        "guidance_rescale": float(payload.get("shape_guidance_rescale", 0.5)),
        "rescale_t": float(payload.get("shape_rescale_t", 3.0)),
    }
    tex_params = {
        "steps": int(payload.get("tex_steps", 12)),
        "guidance_strength": float(payload.get("tex_guidance_strength", 1.0)),
        "guidance_rescale": float(payload.get("tex_guidance_rescale", 0.0)),
        "rescale_t": float(payload.get("tex_rescale_t", 3.0)),
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        if len(images) > 1:
            mesh = pipeline.run_multi_image(
                images,
                seed=seed,
                preprocess_image=preprocess_image,
                sparse_structure_sampler_params=sparse_params,
                shape_slat_sampler_params=shape_params,
                tex_slat_sampler_params=tex_params,
                pipeline_type=pipeline_type,
                max_num_tokens=int(payload.get("max_num_tokens", 16384)),
                mode=multi_image_mode,
            )[0]
        else:
            mesh = pipeline.run(
                images[0],
                seed=seed,
                preprocess_image=preprocess_image,
                sparse_structure_sampler_params=sparse_params,
                shape_slat_sampler_params=shape_params,
                tex_slat_sampler_params=tex_params,
                pipeline_type=pipeline_type,
                max_num_tokens=int(payload.get("max_num_tokens", 16384)),
            )[0]
    except Exception as exc:
        if _is_cuda_oom(exc):
            _raise_cuda_oom(exc, "inference (pipeline.run)")
        raise

    resolution = {
        "512": 512,
        "1024": 1024,
        "1024_cascade": 1024,
        "1536_cascade": 1536,
    }[pipeline_type]

    try:
        glb_bytes = mesh_to_glb(
            mesh,
            resolution=resolution,
            decimation_target=int(payload.get("decimation_target", 500000)),
            texture_size=int(payload.get("texture_size", 1024)),
            remesh=bool(payload.get("remesh", False)),
        )
    except Exception as exc:
        if _is_cuda_oom(exc):
            _raise_cuda_oom(exc, "GLB export (mesh_to_glb)")
        raise
    finally:
        del mesh

    os.makedirs(output_dir, exist_ok=True)
    filename = f"trellis2_{uuid.uuid4().hex}.glb"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as f:
        f.write(glb_bytes)

    torch.cuda.empty_cache()

    return {
        "filename": filename,
        "filepath": filepath,
        "format": "glb",
    }


def health() -> dict[str, Any]:
    info: dict[str, Any] = {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pipeline_loaded": _PIPELINE is not None,
        "model_id": _MODEL_ID,
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info["vram_free_gb"] = round(free / 1e9, 2)
        info["vram_total_gb"] = round(total / 1e9, 2)
        info["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
    return info
