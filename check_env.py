#!/usr/bin/env python
"""Environment checker for the structural-break 2026 validation setup.

Run this FIRST on any new (GPU) box before launching validation harnesses:

    python check_env.py

It verifies, without side effects:
  1. Python version
  2. Required third-party packages import + version vs. the pinned baseline
  3. torch: build type, CUDA availability, device name, a tiny GPU tensor op
  4. Data files present in ./data (X_train / y_train / y_train_index)
  5. Local harness modules import cleanly (_exp_fast, _overnight, _ensemble_tune)

Exit code 0 = all critical checks passed, 1 = at least one FAIL.
WARN items (e.g. version drift, CPU-only torch) do not fail the run but are
flagged because they can change results or defeat the point of a GPU box.
"""
from __future__ import annotations

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# import name -> (pip name, expected version on the reference training box)
PACKAGES = {
    "numpy": ("numpy", "2.5.1"),
    "pandas": ("pandas", "3.0.5"),
    "sklearn": ("scikit-learn", "1.9.0"),
    "scipy": ("scipy", "1.18.0"),
    "joblib": ("joblib", "1.5.3"),
    "threadpoolctl": ("threadpoolctl", "3.6.0"),
    "lightgbm": ("lightgbm", "4.7.0"),
    "matplotlib": ("matplotlib", "3.11.1"),
}

DATA_FILES = ["X_train.parquet", "y_train.parquet", "y_train_index.parquet"]
HARNESS_MODULES = ["_exp_fast", "_overnight", "_ensemble_tune"]

GREEN, YELLOW, RED, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[0m"
_fail = 0
_warn = 0


def ok(msg: str) -> None:
    print(f"  {GREEN}PASS{RESET}  {msg}")


def warn(msg: str) -> None:
    global _warn
    _warn += 1
    print(f"  {YELLOW}WARN{RESET}  {msg}")


def fail(msg: str) -> None:
    global _fail
    _fail += 1
    print(f"  {RED}FAIL{RESET}  {msg}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_python() -> None:
    section("Python")
    v = sys.version_info
    vs = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 13):
        ok(f"Python {vs}")
    elif v.major == 3 and v.minor >= 11:
        warn(f"Python {vs} (reference box is 3.13.7; minor drift usually fine)")
    else:
        fail(f"Python {vs} is too old (need >= 3.11, reference 3.13.7)")
    print(f"        executable: {sys.executable}")


def check_packages() -> None:
    section("Packages")
    for import_name, (pip_name, expected) in PACKAGES.items():
        try:
            mod = importlib.import_module(import_name)
        except Exception as exc:  # noqa: BLE001
            fail(f"{pip_name}: import failed ({exc!r})  ->  pip install {pip_name}=={expected}")
            continue
        got = getattr(mod, "__version__", "?")
        if got == expected:
            ok(f"{pip_name}=={got}")
        else:
            warn(f"{pip_name}=={got}  (reference {expected}; results may differ)")


def check_torch() -> None:
    section("Torch / GPU")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        fail(f"torch import failed ({exc!r})  ->  pip install torch==2.13.0")
        return
    build = torch.__version__
    print(f"        torch {build}  (CUDA build: {torch.version.cuda})")
    if "+cpu" in build or torch.version.cuda is None:
        warn("this is a CPU-only torch build — GPU validation will run on CPU. "
             "Install the CUDA wheel: pip install torch==2.13.0")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        n = torch.cuda.device_count()
        ok(f"CUDA available — {n} device(s), primary: {name}")
        try:
            x = torch.randn(1024, 1024, device="cuda")
            y = (x @ x).sum().item()
            torch.cuda.synchronize()
            ok(f"GPU tensor op succeeded (checksum {y:.1f})")
        except Exception as exc:  # noqa: BLE001
            fail(f"GPU tensor op failed ({exc!r})")
    else:
        warn("torch.cuda.is_available() == False — no GPU visible to torch")


def check_data() -> None:
    section("Data files (./data)")
    data_dir = os.path.join(HERE, "data")
    if not os.path.isdir(data_dir):
        fail(f"missing directory {data_dir}  ->  copy data/ or run `crunch download`")
        return
    for fn in DATA_FILES:
        p = os.path.join(data_dir, fn)
        if os.path.isfile(p):
            mb = os.path.getsize(p) / 1e6
            ok(f"{fn} ({mb:.1f} MB)")
        else:
            fail(f"missing {fn}  ->  run `crunch download` or copy from training box")


def check_harness() -> None:
    section("Harness modules")
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    # keep BLAS single-threaded during import, matching the run-time convention
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for name in HARNESS_MODULES:
        try:
            importlib.import_module(name)
            ok(f"import {name}")
        except Exception as exc:  # noqa: BLE001
            fail(f"import {name} failed ({exc!r})")


def main() -> int:
    print("Structural-break 2026 — environment check")
    print(f"cwd: {os.getcwd()}")
    check_python()
    check_packages()
    check_torch()
    check_data()
    check_harness()

    section("Summary")
    if _fail:
        print(f"  {RED}{_fail} FAIL{RESET}, {YELLOW}{_warn} WARN{RESET} — fix FAILs before validating.")
        return 1
    if _warn:
        print(f"  {GREEN}0 FAIL{RESET}, {YELLOW}{_warn} WARN{RESET} — usable; review WARNs (esp. GPU/version drift).")
        return 0
    print(f"  {GREEN}All checks passed.{RESET} Ready to validate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
