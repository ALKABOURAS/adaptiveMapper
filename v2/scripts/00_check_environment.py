#!/usr/bin/env python3
"""Check what is installed and report exactly what can and cannot run.

Run this before anything else. It uses only the standard library, so it works
even when nothing else is installed yet.

    python v2/scripts/00_check_environment.py
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

#: Newest interpreter for which the scientific stack reliably ships Windows
#: wheels. Beyond this, pip falls back to compiling from source and needs the
#: Visual Studio C++ toolchain -- which is the contourpy failure.
SAFE_MAX_MINOR = 13


def _found(module: str) -> tuple[bool, str]:
    spec = importlib.util.find_spec(module)
    if spec is None:
        return False, "not installed"
    try:
        version = importlib.import_module(module).__version__
    except Exception:
        version = "unknown version"
    return True, version


def main() -> int:
    print("=" * 68)
    print("adaptiveMapper v2 -- environment check")
    print("=" * 68)

    major, minor = sys.version_info[:2]
    print(f"  Python      : {platform.python_version()} ({platform.machine()})")
    print(f"  Interpreter : {sys.executable}")

    in_venv = sys.prefix != sys.base_prefix
    print(f"  Virtualenv  : {'yes' if in_venv else 'NO -- activate .venv first'}")
    print()

    too_new = (major, minor) > (3, SAFE_MAX_MINOR)
    if too_new:
        print(f"  ! Python 3.{minor} is newer than the scientific stack reliably")
        print(f"    ships Windows wheels for (3.{SAFE_MAX_MINOR} and below).")
        print("    This is the cause of the contourpy/meson build error: with no")
        print("    wheel available, pip tries to compile from C source and needs")
        print("    Visual Studio Build Tools.")
        print()
        print(f"    Best fix: create the venv with Python 3.12 or 3.{SAFE_MAX_MINOR}.")
        print()

    print("  Packages")
    print("  " + "-" * 64)
    packages = {
        "numpy": "required -- all analysis",
        "matplotlib": "optional -- figures only",
        "hid": "optional -- Joy-Con hardware only",
        "pytest": "optional -- tests only",
    }
    status = {}
    for module, purpose in packages.items():
        ok, detail = _found(module)
        status[module] = ok
        mark = "OK  " if ok else "MISS"
        print(f"  [{mark}] {module:<12} {detail:<18} {purpose}")
    print()

    print("  What you can run right now")
    print("  " + "-" * 64)
    can = []
    cannot = []

    if status["numpy"]:
        can.append("scripts/03_evaluate_filters.py   (full Level 1 comparison)")
        can.append("all filter and analysis code, synthetic data, ground truth")
    else:
        cannot.append("everything -- numpy is the one hard requirement")

    if status["numpy"] and status["pytest"]:
        can.append("pytest tests/                    (22 tests)")
    elif status["numpy"]:
        cannot.append("pytest tests/                 -- pip install pytest")

    if status["numpy"] and status["matplotlib"]:
        can.append("figure generation in the verification script")
    elif status["numpy"]:
        cannot.append("figures                       -- matplotlib missing (tables still print)")

    if status["hid"]:
        can.append("scripts/01_verify_acquisition.py (needs the controller paired)")
    else:
        cannot.append("scripts/01_verify_acquisition.py -- hidapi missing")

    for line in can:
        print(f"    yes  {line}")
    for line in cannot:
        print(f"    no   {line}")
    print()

    if not status["numpy"]:
        print("  Next step")
        print("  " + "-" * 64)
        print("    pip install --only-binary=:all: numpy")
        print()
        print("  Note the --only-binary flag: it stops pip from falling back to")
        print("  a source build when no wheel matches your interpreter.")
        return 1

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    try:
        from am.filters import build_registry

        registry = build_registry()
        print(f"  Package imports OK -- {len(registry)} filter conditions available:")
        print(f"    {' '.join(registry)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ! Package import FAILED: {exc}")
        return 1

    print()
    print("  Ready. Run: python scripts/03_evaluate_filters.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
