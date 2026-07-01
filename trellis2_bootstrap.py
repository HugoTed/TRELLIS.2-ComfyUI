"""Bootstrap TRELLIS.2 worker venv and CUDA extensions."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

from trellis2_config import get_plugin_root, get_venv_dir, get_worker_python, load_config

_REQUIRED_IMPORTS = (
    "o_voxel",
    "cumesh",
    "nvdiffrast.torch",
    "flex_gemm",
    "nvdiffrec_render",
)


def _run(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    *,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(f"[TRELLIS.2 Bootstrap] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
        capture_output=capture,
        text=True,
    )
    if check and result.returncode != 0:
        if capture:
            raise RuntimeError(_format_subprocess_output(result))
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)
    return result


def _format_subprocess_output(result: subprocess.CompletedProcess[str], *, max_lines: int = 40) -> str:
    chunks: List[str] = []
    for label, stream in (("stdout", result.stdout), ("stderr", result.stderr)):
        if not stream:
            continue
        lines = stream.strip().splitlines()
        if len(lines) > max_lines:
            lines = ["..."] + lines[-max_lines:]
        chunks.append(f"{label}:\n" + "\n".join(lines))
    return "\n\n".join(chunks) if chunks else "(no output captured)"


def _format_pip_failure(cmd: List[str], result: subprocess.CompletedProcess[str]) -> str:
    message = (
        f"pip install failed (exit {result.returncode}): {' '.join(cmd)}\n"
        f"{_format_subprocess_output(result)}"
    )
    combined = f"{result.stdout or ''}\n{result.stderr or ''}"
    if "mathcalls.h" in combined and "cospi" in combined:
        message += (
            "\n\nHint: glibc 2.38+ (Ubuntu 25.04/26.04) needs a one-time CUDA header patch:\n"
            "  sudo python scripts/patch_cuda_math_functions.py"
        )
    if "the global scope has no" in combined and "cmath" in combined:
        message += (
            "\n\nHint: unset CPATH/CPLUS_INCLUDE_PATH if set, then patch CUDA headers:\n"
            "  sudo python3 scripts/patch_cuda_math_functions.py"
        )
    if "cusparse.h" in combined:
        message += (
            "\n\nHint: install CUDA library headers (PyTorch extensions need cuSPARSE etc.):\n"
            "  sudo apt install cuda-libraries-dev-12-8"
        )
    if "cannot find -lcuda" in combined:
        message += (
            "\n\nHint: add CUDA driver stub libs for linking, e.g.:\n"
            "  export LIBRARY_PATH=/usr/local/cuda-12.8/lib64/stubs:$LIBRARY_PATH\n"
            "Or: sudo apt install cuda-driver-dev-12-8"
        )
    return message


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


def _pip_install(
    worker_python: Path,
    args: List[str],
    cwd: Optional[Path] = None,
    *,
    required: bool = True,
    env: Optional[dict] = None,
) -> None:
    cmd = [str(worker_python), "-m", "pip", "install", *args]
    print(f"[TRELLIS.2 Bootstrap] {' '.join(cmd)}")
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=run_env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = _format_pip_failure(cmd, result)
        if required:
            raise RuntimeError(message)
        print(f"[TRELLIS.2 Bootstrap] Warning: {message}")


def _compiler_major_version(compiler: str) -> Optional[int]:
    try:
        result = subprocess.run(
            [compiler, "-dumpversion"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip().split(".")[0])
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _nvcc_version(cuda_home: Path) -> Optional[tuple[int, int]]:
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        return None
    try:
        result = subprocess.run(
            [str(nvcc), "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        for token in result.stdout.replace(",", " ").split():
            if token.count(".") >= 1 and token[0].isdigit():
                parts = token.split(".")
                return int(parts[0]), int(parts[1])
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None
    return None


def _resolve_cuda_toolkit() -> dict:
    """Prefer the newest installed CUDA toolkit (12.8+ avoids glibc header bugs on Ubuntu 26.04)."""
    if os.environ.get("CUDA_HOME"):
        cuda_home = Path(os.environ["CUDA_HOME"])
        if (cuda_home / "bin" / "nvcc").is_file():
            bin_dir = str(cuda_home / "bin")
            path = os.environ.get("PATH", "")
            if not path.startswith(bin_dir):
                path = bin_dir + os.pathsep + path
            return {"CUDA_HOME": str(cuda_home), "PATH": path}

    for version in ("13.0", "12.8", "12.6", "12.5", "12.4", "12.3"):
        cuda_home = Path(f"/usr/local/cuda-{version}")
        if (cuda_home / "bin" / "nvcc").is_file():
            print(f"[TRELLIS.2 Bootstrap] Using CUDA toolkit: {cuda_home}")
            return {
                "CUDA_HOME": str(cuda_home),
                "PATH": f"{cuda_home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
            }

    default = Path("/usr/local/cuda")
    if (default / "bin" / "nvcc").is_file():
        return {
            "CUDA_HOME": str(default),
            "PATH": f"{default / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        }
    return {}


def _glibc_version() -> Optional[tuple[int, int]]:
    for cmd in (["ldd", "--version"],):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            match = re.search(r"GLIBC (\d+)\.(\d+)", result.stdout)
            if match:
                return int(match.group(1)), int(match.group(2))
        except (OSError, subprocess.CalledProcessError):
            pass
    try:
        libc, version = platform.libc_ver()
        if libc == "glibc" and version:
            parts = version.split(".")
            return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        pass
    return None


def _needs_glibc_cuda_compat() -> bool:
    if os.name == "nt":
        return False
    glibc = _glibc_version()
    return glibc is not None and glibc >= (2, 38)


def _ensure_cuda_math_header_patch() -> None:
    """Patch CUDA math_functions.h on glibc >= 2.38 (shim breaks cmath; patch is required)."""
    if not _needs_glibc_cuda_compat():
        return

    plugin_root = get_plugin_root()
    script = plugin_root / "scripts" / "patch_cuda_math_functions.py"
    if not script.is_file():
        raise RuntimeError(f"Missing CUDA patch script: {script}")

    glibc = _glibc_version()
    print(f"[TRELLIS.2 Bootstrap] Checking CUDA header patch for glibc {glibc[0]}.{glibc[1]}...")
    result = subprocess.run(
        [sys.executable, str(script), "--ensure"],
        cwd=str(plugin_root),
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "CUDA header patch failed").strip()
        raise RuntimeError(detail)


def _max_gcc_major_for_cuda(cuda_version: Optional[tuple[int, int]]) -> int:
    if cuda_version is None:
        return 13
    major, minor = cuda_version
    if major > 12 or (major == 12 and minor >= 6):
        return 14
    if major == 12 and minor >= 4:
        return 13
    return 12


def _append_cuda_link_paths(env: dict, cuda_home: Optional[Path]) -> dict:
    if cuda_home is None:
        return env
    lib64 = cuda_home / "lib64"
    stubs = lib64 / "stubs"
    paths: List[str] = []
    if stubs.is_dir():
        paths.append(str(stubs))
    if lib64.is_dir():
        paths.append(str(lib64))
    if not paths:
        return env
    joined = os.pathsep.join(paths)
    for key in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        env[key] = joined + os.pathsep + env.get(key, os.environ.get(key, ""))
    link_flags = env.get("LDFLAGS", os.environ.get("LDFLAGS", ""))
    for path in paths:
        link_flags = f"-L{path} {link_flags}".strip()
    env["LDFLAGS"] = link_flags
    return env


def _cuda_build_env() -> dict:
    """Pick CUDA toolkit + host compiler versions that can build extensions."""
    env = _resolve_cuda_toolkit()
    cuda_home = Path(env["CUDA_HOME"]) if env.get("CUDA_HOME") else None
    cuda_version = _nvcc_version(cuda_home) if cuda_home else None
    max_gcc = _max_gcc_major_for_cuda(cuda_version)

    if cuda_version == (12, 4):
        print(
            "[TRELLIS.2 Bootstrap] Warning: CUDA 12.4 may fail on Ubuntu 24.04+/26.04. "
            "Prefer cuda-nvcc-12-8 if builds fail."
        )

    cc = os.environ.get("CC")
    cxx = os.environ.get("CXX")
    if cc and cxx:
        major = _compiler_major_version(cxx)
        if major is not None and major <= max_gcc:
            env.update({"CC": cc, "CXX": cxx})
        else:
            print(
                f"[TRELLIS.2 Bootstrap] Warning: {cxx} is GCC {major}; "
                f"CUDA {cuda_version or 'toolkit'} needs GCC <= {max_gcc}. Searching..."
            )
            cc = cxx = None

    if not env.get("CC"):
        for version in range(max_gcc, 10, -1):
            gcc = shutil.which(f"gcc-{version}")
            gxx = shutil.which(f"g++-{version}")
            if gcc and gxx:
                print(f"[TRELLIS.2 Bootstrap] Using host compiler: {gcc}, {gxx}")
                env.update({"CC": gcc, "CXX": gxx})
                break

    if not env.get("CC"):
        default_gxx = shutil.which("g++") or shutil.which("c++")
        major = _compiler_major_version(default_gxx) if default_gxx else None
        if major is not None and major > max_gcc:
            raise RuntimeError(
                f"Default GCC {major} is too new for CUDA toolkit "
                f"{'.'.join(map(str, cuda_version)) if cuda_version else 'nvcc'} "
                f"(requires GCC <= {max_gcc}).\n"
                f"Install a compatible compiler, e.g.:\n"
                f"  sudo apt install gcc-{max_gcc} g++-{max_gcc}\n"
                f"Or set: export CC=gcc-{max_gcc} CXX=g++-{max_gcc}"
            )

    env = _append_cuda_link_paths(env, cuda_home)
    stubs_lib = (cuda_home / "lib64" / "stubs" / "libcuda.so") if cuda_home else None
    if stubs_lib is not None and stubs_lib.is_file():
        print(f"[TRELLIS.2 Bootstrap] Using CUDA link stubs: {stubs_lib.parent}")
    elif cuda_home is not None:
        print(
            "[TRELLIS.2 Bootstrap] Warning: libcuda.so stub not found. "
            "Install: sudo apt install cuda-driver-dev-12-8"
        )
    return env


def _cuda_include_dirs() -> List[Path]:
    env = _resolve_cuda_toolkit()
    dirs: List[Path] = []
    if env.get("CUDA_HOME"):
        dirs.append(Path(env["CUDA_HOME"]) / "include")
    for version in ("13.0", "12.8", "12.6", "12.5", "12.4"):
        dirs.append(Path(f"/usr/local/cuda-{version}") / "include")
    dirs.append(Path("/usr/local/cuda/include"))
    return dirs


def _cuda_dev_headers_present() -> bool:
    for include_dir in _cuda_include_dirs():
        if (include_dir / "cusparse.h").is_file():
            return True
    return False


def _check_cuda_build_prereqs() -> None:
    issues: List[str] = []
    cuda_env = _resolve_cuda_toolkit()
    if not cuda_env and shutil.which("nvcc") is None:
        issues.append(
            "nvcc not found. Install CUDA build packages, e.g.:\n"
            "  sudo apt install cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-libraries-dev-12-8\n"
            "  export CUDA_HOME=/usr/local/cuda-12.8\n"
            "  export PATH=/usr/local/cuda-12.8/bin:$PATH"
        )
    elif not _cuda_dev_headers_present():
        issues.append(
            "CUDA library headers missing (cusparse.h not found). Install:\n"
            "  sudo apt install cuda-libraries-dev-12-8"
        )
    if shutil.which("g++") is None and shutil.which("c++") is None:
        issues.append("C++ compiler not found. On Debian/Ubuntu: sudo apt install build-essential")
    if issues:
        raise RuntimeError(
            "CUDA extension build prerequisites are missing:\n- " + "\n- ".join(issues)
        )


def _ensure_ovoxel_source(plugin_root: Path) -> Path:
    ovoxel_src = plugin_root / "o-voxel"
    eigen_marker = ovoxel_src / "third_party" / "eigen" / "Eigen"
    if not ovoxel_src.is_dir():
        raise RuntimeError(
            "o-voxel source directory not found. Clone the plugin with:\n"
            "  git clone --recursive https://github.com/your/TRELLIS.2-ComfyUI.git"
        )
    if eigen_marker.is_dir():
        return ovoxel_src

    if (plugin_root / ".git").exists():
        print("[TRELLIS.2 Bootstrap] Initializing git submodules for o-voxel...")
        _run(["git", "submodule", "update", "--init", "--recursive"], cwd=plugin_root)

    if not eigen_marker.is_dir():
        raise RuntimeError(
            "o-voxel Eigen submodule is missing. From the plugin directory run:\n"
            "  git submodule update --init --recursive"
        )
    return ovoxel_src


def verify_worker_imports(worker_python: Path) -> None:
    modules = ", ".join(repr(name) for name in _REQUIRED_IMPORTS)
    script = f"""
import importlib
missing = []
for name in [{modules}]:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{{name}}: {{exc}}")
if missing:
    raise SystemExit("Missing required worker modules:\\n" + "\\n".join(missing))
"""
    result = subprocess.run(
        [str(worker_python), "-c", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "Worker environment is missing required CUDA extension packages.\n"
            f"{detail}\n"
            "Re-run TRELLIS.2 Setup with force_reinstall=true after fixing build prerequisites."
        )


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
    _check_cuda_build_prereqs()
    _ensure_cuda_math_header_patch()
    build_env = _cuda_build_env()
    tmp = Path(tempfile.mkdtemp(prefix="trellis2-ext-"))

    extensions = [
        ("nvdiffrast", "https://github.com/NVlabs/nvdiffrast.git", "v0.4.0", False),
        ("nvdiffrec", "https://github.com/JeffreyXiang/nvdiffrec.git", "renderutils", False),
        ("CuMesh", "https://github.com/JeffreyXiang/CuMesh.git", None, True),
        ("FlexGEMM", "https://github.com/JeffreyXiang/FlexGEMM.git", None, True),
    ]

    try:
        for name, url, branch, recursive in extensions:
            dest = tmp / name
            print(f"[TRELLIS.2 Bootstrap] Installing {name}...")
            _clone_repo(url, dest, branch=branch, recursive=recursive)
            _pip_install(
                worker_python,
                [str(dest), "--no-build-isolation"],
                cwd=plugin_root,
                env=build_env,
            )

        ovoxel_src = _ensure_ovoxel_source(plugin_root)
        print("[TRELLIS.2 Bootstrap] Installing o-voxel from local source...")
        _pip_install(
            worker_python,
            [str(ovoxel_src), "--no-build-isolation"],
            cwd=plugin_root,
            env=build_env,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("[TRELLIS.2 Bootstrap] Installing flash-attn (optional)...")
    _pip_install(
        worker_python,
        ["flash-attn==2.7.3", "--no-build-isolation"],
        cwd=plugin_root,
        required=False,
        env=build_env,
    )
    if subprocess.run(
        [str(worker_python), "-c", "import flash_attn"],
        capture_output=True,
    ).returncode != 0:
        print("[TRELLIS.2 Bootstrap] flash-attn not installed; set ATTN_BACKEND=sdpa or install xformers.")

    verify_worker_imports(worker_python)


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
        worker_python = get_worker_python()
        if worker_python is not None:
            verify_worker_imports(worker_python)

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
