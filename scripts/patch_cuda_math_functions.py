#!/usr/bin/env python3
"""Patch CUDA math_functions.h for glibc >= 2.38 (Ubuntu 25.04+/26.04).

Adds noexcept(true) to sinpi/cospi/rsqrt declarations so nvcc can compile on
modern glibc. Idempotent — safe to run multiple times.

Usage:
  python scripts/patch_cuda_math_functions.py
  sudo python scripts/patch_cuda_math_functions.py   # if CUDA headers are root-owned
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

PATCHES = (
    ("double", "sinpi"),
    ("float", "sinpif"),
    ("double", "cospi"),
    ("float", "cospif"),
    ("double", "rsqrt"),
    ("float", "rsqrtf"),
)

PATCHED_MARKER = "sinpi(double x) noexcept"


def find_math_functions_paths() -> list[Path]:
    roots: list[Path] = []
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        roots.append(Path(cuda_home))
    for version in ("13.0", "12.8", "12.6", "12.5", "12.4"):
        roots.append(Path(f"/usr/local/cuda-{version}"))
    roots.append(Path("/usr/local/cuda"))

    found: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        for rel in (
            "targets/x86_64-linux/include/crt/math_functions.h",
            "include/crt/math_functions.h",
        ):
            path = root / rel
            key = str(path.resolve()) if path.exists() else str(path)
            if path.is_file() and key not in seen:
                seen.add(key)
                found.append(path)
    return found


def is_patched(path: Path) -> bool:
    try:
        return PATCHED_MARKER in path.read_text(encoding="utf-8")
    except OSError:
        return False


def patch_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    if PATCHED_MARKER in text:
        print(f"Already patched: {path}")
        return 0

    original = text
    for ctype, name in PATCHES:
        pattern = (
            rf"(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ {ctype}\s+{name}\([^)]*\))"
            r"(?!\s*noexcept)"
        )
        text, count = re.subn(pattern, r"\1 noexcept (true)", text, count=1)
        if count == 0:
            print(f"  warning: could not patch {name} in {path}", file=sys.stderr)

    if text == original:
        print(f"No changes made: {path}")
        return 0

    path.write_text(text, encoding="utf-8")
    print(f"Patched: {path}")
    return 1


def ensure_patched() -> None:
    paths = find_math_functions_paths()
    if not paths:
        raise RuntimeError(
            "Could not find CUDA crt/math_functions.h. Set CUDA_HOME, e.g.:\n"
            "  export CUDA_HOME=/usr/local/cuda-12.8"
        )

    pending = [path for path in paths if not is_patched(path)]
    if not pending:
        print(f"[CUDA patch] math_functions.h already patched ({paths[0]})")
        return

    patched_any = False
    permission_errors: list[Path] = []
    for path in pending:
        try:
            patched_any |= patch_file(path) > 0
        except PermissionError:
            permission_errors.append(path)

    if permission_errors:
        target = permission_errors[0]
        raise RuntimeError(
            "Ubuntu 25.04+/26.04 requires a one-time CUDA header patch (glibc compatibility).\n"
            f"Permission denied: {target}\n"
            "Run once from the plugin directory:\n"
            "  sudo python scripts/patch_cuda_math_functions.py"
        )

    if not patched_any:
        raise RuntimeError(
            "CUDA math_functions.h patch did not apply. "
            "Run manually: sudo python scripts/patch_cuda_math_functions.py"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ensure",
        action="store_true",
        help="Patch if needed; exit non-zero on failure (for bootstrap).",
    )
    args = parser.parse_args()

    try:
        if args.ensure:
            ensure_patched()
        else:
            paths = find_math_functions_paths()
            if not paths:
                print("Could not find CUDA crt/math_functions.h. Set CUDA_HOME.", file=sys.stderr)
                return 1
            for path in paths:
                try:
                    patch_file(path)
                except PermissionError:
                    print(f"Permission denied: {path}", file=sys.stderr)
                    print(
                        "Re-run with sudo, e.g.: sudo python scripts/patch_cuda_math_functions.py",
                        file=sys.stderr,
                    )
                    return 1
        return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
