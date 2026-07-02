"""Persistent HTTP worker for TRELLIS.2 — runs in isolated .trellis2-venv."""

from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

PLUGIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, PLUGIN_ROOT)

os.chdir(PLUGIN_ROOT)
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from trellis2.utils.attn_env import ensure_attn_backend_env  # noqa: E402

ensure_attn_backend_env()

from trellis2_config import load_config  # noqa: E402
from worker import inference  # noqa: E402

_config = load_config()
WORKER_HOST = _config["worker"]["host"]
WORKER_PORT = _config["worker"]["port"]


def _resolve_output_dir() -> str:
    output_dir = os.environ.get("TRELLIS2_OUTPUT_DIR")
    if output_dir:
        return output_dir
    default = os.path.join(PLUGIN_ROOT, "output", "comfyui")
    os.makedirs(default, exist_ok=True)
    return default


class Trellis2WorkerHandler(BaseHTTPRequestHandler):
    output_dir: str = ""

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        sys.stderr.write("[trellis2-worker] " + (format % args) + "\n")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, inference.health())
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        if path == "/generate":
            try:
                result = inference.generate_glb(payload, self.output_dir)
                self._send_json(200, {"status": "ok", **result})
            except Exception as exc:
                traceback.print_exc()
                self._send_json(500, {"status": "error", "error": str(exc)})
            return

        if path == "/preload":
            try:
                model_id = payload.get("model_id", "microsoft/TRELLIS.2-4B")
                inference.get_pipeline(model_id)
                self._send_json(200, {"status": "ok", "model_id": model_id})
            except Exception as exc:
                traceback.print_exc()
                self._send_json(500, {"status": "error", "error": str(exc)})
            return

        self._send_json(404, {"error": "not found"})


def main() -> None:
    output_dir = _resolve_output_dir()
    Trellis2WorkerHandler.output_dir = output_dir

    server = ThreadingHTTPServer((WORKER_HOST, WORKER_PORT), Trellis2WorkerHandler)
    print(f"[trellis2-worker] plugin root: {PLUGIN_ROOT}", flush=True)
    print(f"[trellis2-worker] listening on http://{WORKER_HOST}:{WORKER_PORT}", flush=True)
    print(f"[trellis2-worker] output dir: {output_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[trellis2-worker] shutting down", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
