#!/usr/bin/env python3
"""Patch CUDA math_functions.h for glibc >= 2.38 (Ubuntu 25.04+/26.04).

Adds noexcept(true) to sinpi/cospi/rsqrt declarations so nvcc can compile on
modern glibc. Idempotent — safe to run multiple times.

Usage:
  python scripts/patch_cuda_math_functions.py
  sudo python scripts/patch_cuda_math_functions.py   # if CUDA headers are root-owned
"""

from __future__ import annotations

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


def _find_math_functions_h() -> Path | None:
    candidates: list[Path] = []
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidates.append(Path(cuda_home))
    for version in ("13.0", "12.8", "12.6", "12.5", "12.4"):
        candidates.append(Path(f"/usr/local/cuda-{version}"))
    candidates.append(Path("/usr/local/cuda"))

    for root in candidates:
        direct = root / "include" / "crt" / "math_functions.h"
        targets = root / "targets" / "x86_64-linux" / "include" / "crt" / "math_functions.h"
        for path in (targets, direct):
            if path.is_file():
                return path
    return None


def patch_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    original = text
    for ctype, name in PATCHES:
        pattern = (
            rf"(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ {ctype}\s+{name}\([^)]*\))"
            r"(?!\s*noexcept)"
        )
        text, count = re.subn(pattern, r"\1 noexcept (true)", text, count=1)
        if count == 0:
            print(f"  skip (already patched or not found): {name}", file=sys.stderr)

    if text == original:
        print(f"No changes needed: {path}")
        return 0

    path.write_text(text, encoding="utf-8")
    print(f"Patched: {path}")
    return 1


def main() -> int:
    path = _find_math_functions_h()
    if path is None:
        print("Could not find CUDA crt/math_functions.h. Set CUDA_HOME.", file=sys.stderr)
        return 1
    try:
        return 0 if patch_file(path) == 0 else 0
    except PermissionError:
        print(f"Permission denied: {path}", file=sys.stderr)
        print("Re-run with sudo, e.g.: sudo python scripts/patch_cuda_math_functions.py", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
