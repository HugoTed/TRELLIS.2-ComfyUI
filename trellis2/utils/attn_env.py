"""Select attention backend before trellis2 attention modules import."""

from __future__ import annotations

import importlib.util
import os


def ensure_attn_backend_env() -> str:
    """Set ATTN_BACKEND when unset, preferring flash-attn then xformers then PyTorch SDPA."""
    existing = os.environ.get("ATTN_BACKEND")
    if existing:
        return existing

    if importlib.util.find_spec("flash_attn") is not None:
        backend = "flash_attn"
    elif importlib.util.find_spec("xformers") is not None:
        backend = "xformers"
    else:
        backend = "sdpa"

    os.environ["ATTN_BACKEND"] = backend
    print(f"[TRELLIS.2] ATTN_BACKEND not set; using {backend}")
    return backend
