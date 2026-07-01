"""Plugin configuration (no heavy dependencies)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

PLUGIN_ROOT = Path(__file__).resolve().parent
VENV_DIR_NAME = ".trellis2-venv"
WORKER_HOST = "127.0.0.1"
WORKER_PORT = 18188

DEFAULTS: Dict[str, Any] = {
    "worker": {
        "host": WORKER_HOST,
        "port": WORKER_PORT,
        "venv": VENV_DIR_NAME,
        "auto_start": True,
        "auto_install": True,
        "startup_timeout": 600,
    },
    "model": {
        "auto_download": True,
        "model_id": "microsoft/TRELLIS.2-4B",
    },
    "torch": {
        "index_url": "https://download.pytorch.org/whl/cu124",
        "version": "2.6.0",
        "torchvision_version": "0.21.0",
    },
}


def get_plugin_root() -> Path:
    return PLUGIN_ROOT


def get_config_path() -> Path:
    env_path = os.environ.get("TRELLIS2_CONFIG")
    if env_path:
        return Path(env_path)
    return PLUGIN_ROOT / "config.json"


def load_config() -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULTS))
    config_path = get_config_path()
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            user_config = json.load(f)
        _deep_merge(config, user_config)
    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get_venv_dir(config: Dict[str, Any] | None = None) -> Path:
    config = config or load_config()
    venv_name = config["worker"]["venv"]
    venv_path = Path(venv_name)
    if not venv_path.is_absolute():
        venv_path = PLUGIN_ROOT / venv_path
    return venv_path


def get_worker_python(config: Dict[str, Any] | None = None) -> Path | None:
    env_python = os.environ.get("TRELLIS2_PYTHON")
    if env_python:
        path = Path(env_python)
        if path.is_file():
            return path

    venv_dir = get_venv_dir(config)
    if os.name == "nt":
        candidates = [venv_dir / "Scripts" / "python.exe"]
    else:
        candidates = [venv_dir / "bin" / "python", venv_dir / "bin" / "python3"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def get_worker_url(config: Dict[str, Any] | None = None) -> str:
    config = config or load_config()
    host = config["worker"]["host"]
    port = config["worker"]["port"]
    return f"http://{host}:{port}"


def get_model_id(config: Dict[str, Any] | None = None) -> str:
    config = config or load_config()
    return config["model"]["model_id"]
