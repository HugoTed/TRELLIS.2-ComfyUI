"""Bootstrap TRELLIS.2 worker venv and CUDA extensions."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

from config import get_plugin_root, get_venv_dir, get_worker_python, load_config


def _run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    print(f"[TRELLIS.2 Bootstrap] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def find_python310() -> Optional[Path]:
    env_python = os.environ.get("TRELLIS2_PYTHON")
    if env_python and Path(env_python).is_file():
        return Path(env_python)

    if shutil.which("uv"):
        try:
            result = subprocess.run(
                ["uv", "python", "find", "3.10"],
                capture_output=True,
                text=True,
                check=True,
            )
            candidate = result.stdout.strip()
            if candidate and Path(candidate).is_file():
                return Path(candidate)
        except subprocess.CalledProcessError:
            pass

    if os.name == "nt" and shutil.which("py"):
        for version in ("3.10", "3.11", "3.12"):
            try:
                result = subprocess.run(
                    ["py", f"-{version}", "-c", "import sys; print(sys.executable)"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                candidate = result.stdout.strip()
                if candidate and Path(candidate).is_file():
                    return Path(candidate)
            except subprocess.CalledProcessError:
                continue

    for name in ("python3.10", "python3", "python"):
        candidate = shutil.which(name)
        if not candidate:
            continue
        try:
            result = subprocess.run(
                [candidate, "-c", "import sys; assert sys.version_info[:2] >= (3, 10)"],
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                return Path(candidate)
        except OSError:
            continue
    return None


def venv_exists() -> bool:
    return get_worker_python() is not None


def create_venv() -> Path:
    plugin_root = get_plugin_root()
    venv_dir = get_venv_dir()
    existing = get_worker_python()
    if existing is not None:
        return existing

    python310 = find_python310()
    if python310 is None:
        raise RuntimeError(
            "Python 3.10+ is required for the TRELLIS.2 worker. "
            "Install Python 3.10 or set TRELLIS2_PYTHON to its executable."
        )

    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        if shutil.which("uv"):
            _run(["uv", "venv", str(venv_dir), "--python", str(python310)], cwd=plugin_root)
        else:
            _run([str(python310), "-m", "venv", str(venv_dir)], cwd=plugin_root)
    except subprocess.CalledProcessError as exc:
        hint = ""
        if os.name != "nt":
            hint = " On Debian/Ubuntu: sudo apt install python3.10 python3.10-venv"
        raise RuntimeError(f"Failed to create venv at {venv_dir}.{hint}") from exc

    worker_python = get_worker_python()
    if worker_python is None:
        raise RuntimeError(f"Failed to create worker venv at {venv_dir}")
    return worker_python


def _pip_install(worker_python: Path, args: List[str], cwd: Optional[Path] = None) -> None:
    _run([str(worker_python), "-m", "pip", "install", *args], cwd=cwd)


def _clone_repo(url: str, dest: Path, *, branch: Optional[str] = None, recursive: bool = False) -> None:
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    cmd = ["git", "clone"]
    if recursive:
        cmd.append("--recursive")
    if branch:
        cmd.extend(["-b", branch])
    cmd.extend([url, str(dest)])
    _run(cmd)


def install_cuda_extensions(worker_python: Path) -> None:
    plugin_root = get_plugin_root()
    tmp = Path(tempfile.mkdtemp(prefix="trellis2-ext-"))

    extensions = [
        ("nvdiffrast", "https://github.com/NVlabs/nvdiffrast.git", "v0.4.0", False),
        ("nvdiffrec", "https://github.com/JeffreyXiang/nvdiffrec.git", "renderutils", False),
        ("CuMesh", "https://github.com/JeffreyXiang/CuMesh.git", None, True),
        ("FlexGEMM", "https://github.com/JeffreyXiang/FlexGEMM.git", None, True),
    ]

    for name, url, branch, recursive in extensions:
        dest = tmp / name
        print(f"[TRELLIS.2 Bootstrap] Installing {name}...")
        try:
            _clone_repo(url, dest, branch=branch, recursive=recursive)
            _pip_install(worker_python, [str(dest), "--no-build-isolation"], cwd=plugin_root)
        except subprocess.CalledProcessError as exc:
            print(f"[TRELLIS.2 Bootstrap] Warning: {name} install failed: {exc}")

    # o-voxel from local source
    ovoxel_src = plugin_root / "o-voxel"
    if ovoxel_src.is_dir():
        print("[TRELLIS.2 Bootstrap] Installing o-voxel from local source...")
        try:
            _pip_install(worker_python, [str(ovoxel_src), "--no-build-isolation"], cwd=plugin_root)
        except subprocess.CalledProcessError as exc:
            print(f"[TRELLIS.2 Bootstrap] Warning: o-voxel install failed: {exc}")
    else:
        print("[TRELLIS.2 Bootstrap] Warning: o-voxel directory not found; run git clone --recursive")

    print("[TRELLIS.2 Bootstrap] Installing flash-attn (optional)...")
    try:
        _pip_install(worker_python, ["flash-attn==2.7.3", "--no-build-isolation"], cwd=plugin_root)
    except subprocess.CalledProcessError:
        print("[TRELLIS.2 Bootstrap] flash-attn install failed; set ATTN_BACKEND=sdpa or install xformers.")


def install_worker_dependencies() -> None:
    plugin_root = get_plugin_root()
    worker_python = create_venv()
    config = load_config()
    requirements = plugin_root / "worker" / "requirements.txt"
    torch_index = config["torch"]["index_url"]
    torch_version = config["torch"]["version"]
    torchvision_version = config["torch"]["torchvision_version"]

    _pip_install(worker_python, ["--upgrade", "pip", "setuptools", "wheel"], cwd=plugin_root)
    _pip_install(
        worker_python,
        [
            f"torch=={torch_version}",
            f"torchvision=={torchvision_version}",
            "--index-url",
            torch_index,
        ],
        cwd=plugin_root,
    )
    _pip_install(worker_python, ["-r", str(requirements)], cwd=plugin_root)
    install_cuda_extensions(worker_python)


def ensure_worker_installed(force: bool = False) -> str:
    config = load_config()
    messages: List[str] = []
    venv_dir = get_venv_dir()

    if force and venv_dir.exists():
        messages.append(f"Removing existing worker venv: {venv_dir}")
        shutil.rmtree(venv_dir, ignore_errors=True)

    if force or not venv_exists():
        messages.append("Creating worker virtual environment...")
        create_venv()
        messages.append("Installing worker dependencies (this may take 10–30 minutes)...")
        install_worker_dependencies()
    else:
        messages.append(f"Worker venv ready: {venv_dir}")

    if config["model"].get("auto_download", True):
        messages.append(
            "Model weights download on first inference from Hugging Face "
            f"({config['model']['model_id']})."
        )

    return "\n".join(messages)


def main(argv: Optional[Iterable[str]] = None) -> int:
    force = "--force" in list(argv or sys.argv[1:])
    print(ensure_worker_installed(force=force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
