"""ComfyUI Manager install hook for TRELLIS.2."""

from bootstrap import ensure_worker_installed

if __name__ == "__main__":
    print(ensure_worker_installed())
