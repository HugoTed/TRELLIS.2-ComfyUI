"""HTTP client for the TRELLIS.2 worker (ComfyUI-safe, no torch imports)."""

from __future__ import annotations

import atexit
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from trellis2_bootstrap import ensure_worker_installed, venv_exists
from trellis2_config import get_model_id, get_plugin_root, get_worker_python, get_worker_url, load_config
from trellis2_process_util import popen_detached, terminate_process_tree

_worker_proc = None


def _worker_log_path() -> Path:
    return get_plugin_root() / "trellis2-worker.log"


def _read_log_tail(path: Path, max_lines: int = 50) -> str:
    if not path.is_file():
        return (
            f"No log at {path}. Run manually:\n"
            f"  {get_worker_python() or '<python>'} worker/server.py"
        )
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-max_lines:])
        return f"Last lines from {path}:\n{tail}"
    except OSError as exc:
        return f"Could not read {path}: {exc}"


def _worker_exit_message(exit_code: Optional[int]) -> str:
    log_tail = _read_log_tail(_worker_log_path())
    code = "unknown" if exit_code is None else str(exit_code)
    return (
        f"TRELLIS.2 worker process exited before becoming ready (exit code {code}).\n"
        f"{log_tail}\n"
        "Tip: run `.trellis2-venv/bin/python worker/server.py` in a terminal for full output."
    )


def _resolve_output_dir() -> str:
    try:
        import folder_paths

        return folder_paths.get_output_directory()
    except ImportError:
        out = get_plugin_root() / "output" / "comfyui"
        out.mkdir(parents=True, exist_ok=True)
        return str(out)


def ping_worker(timeout: float = 2.0) -> bool:
    try:
        resp = requests.get(f"{get_worker_url()}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _start_worker_process():
    global _worker_proc
    if _worker_proc is not None and _worker_proc.poll() is None:
        return _worker_proc

    worker_python = get_worker_python()
    if worker_python is None:
        raise RuntimeError("TRELLIS.2 worker venv is not installed. Run trellis2_bootstrap or TRELLIS.2 Setup node.")

    plugin_root = get_plugin_root()
    output_dir = _resolve_output_dir()
    log_path = _worker_log_path()
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"\n--- worker start {datetime.now().isoformat()} ---\n")
    log_file.flush()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(plugin_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["TRELLIS2_OUTPUT_DIR"] = output_dir

    proc = popen_detached(
        [str(worker_python), str(plugin_root / "worker" / "server.py")],
        cwd=str(plugin_root),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    print(f"[TRELLIS.2 Client] worker started (pid={proc.pid}), log: {log_path}")
    _worker_proc = proc

    def cleanup():
        print(f"[TRELLIS.2 Client] terminating worker (pid={proc.pid})")
        terminate_process_tree(proc)

    atexit.register(cleanup)
    return proc


def wait_for_worker(timeout: Optional[float] = None) -> None:
    config = load_config()
    timeout = timeout if timeout is not None else float(config["worker"]["startup_timeout"])
    t0 = time.time()
    while True:
        if ping_worker(timeout=2.0):
            print("[TRELLIS.2 Client] worker is ready")
            return
        proc = _worker_proc
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(_worker_exit_message(proc.returncode))
        if time.time() - t0 > timeout:
            raise RuntimeError(
                f"TRELLIS.2 worker failed to start within {timeout:.0f}s.\n"
                f"{_read_log_tail(_worker_log_path())}\n"
                "Tip: run `.trellis2-venv/bin/python worker/server.py` in a terminal."
            )
        time.sleep(1.0)


def ensure_worker_running(auto_install: Optional[bool] = None) -> str:
    config = load_config()
    auto_install = config["worker"]["auto_install"] if auto_install is None else auto_install
    auto_start = config["worker"]["auto_start"]

    if ping_worker():
        return "TRELLIS.2 worker is already running."

    if auto_install and not venv_exists():
        ensure_worker_installed()

    if not venv_exists():
        raise RuntimeError(
            "TRELLIS.2 worker is not installed. Add TRELLIS.2 Setup node or run: python trellis2_bootstrap.py"
        )

    if not auto_start:
        raise RuntimeError("TRELLIS.2 worker is not running and auto_start is disabled in config.json.")

    _start_worker_process()
    wait_for_worker()
    return "TRELLIS.2 worker started."


def _post_json(path: str, payload: Dict[str, Any], timeout: float = 7200.0) -> Dict[str, Any]:
    ensure_worker_running()
    url = f"{get_worker_url()}{path}"
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Invalid worker response ({resp.status_code}): {resp.text}") from exc
    if resp.status_code >= 400 or data.get("status") == "error":
        raise RuntimeError(data.get("error") or f"Worker error ({resp.status_code})")
    return data


def generate_glb(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload)
    payload.setdefault("model_id", get_model_id())
    return _post_json("/generate", payload)


def preload_model() -> Dict[str, Any]:
    return _post_json("/preload", {"model_id": get_model_id()}, timeout=3600.0)
