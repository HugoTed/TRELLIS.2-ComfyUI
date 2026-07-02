"""Select attention backend before trellis2 attention modules import."""

from __future__ import annotations

import importlib
import os


def _importable(module: str) -> bool:
    """Actually import the module: find_spec alone misses ABI/torch mismatches."""
    try:
        importlib.import_module(module)
        return True
    except Exception as exc:
        print(f"[TRELLIS.2] {module} installed but failed to import ({exc}); skipping")
        return False


def ensure_attn_backend_env() -> str:
    """Set ATTN_BACKEND when unset, preferring flash-attn then xformers then PyTorch SDPA."""
    existing = os.environ.get("ATTN_BACKEND")
    if existing:
        return existing

    if _importable("flash_attn"):
        backend = "flash_attn"
    elif _importable("xformers"):
        backend = "xformers"
    else:
        backend = "sdpa"

    os.environ["ATTN_BACKEND"] = backend
    print(f"[TRELLIS.2] ATTN_BACKEND not set; using {backend}")
    return backend
