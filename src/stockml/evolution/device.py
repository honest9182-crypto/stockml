"""GPU detection for the "xgb" model_family (evolution/model_builder.py).

Every other model family in this codebase (logreg, hgb) is CPU-only via
scikit-learn -- "device" only ever varies for xgb, and even then only
between "cuda" (a usable NVIDIA GPU is present) and "cpu" (the always-safe
fallback, including "no GPU", "no driver", "nvidia-smi missing", or any
other detection failure).
"""

from __future__ import annotations

import functools
import shutil
import subprocess


@functools.lru_cache(maxsize=1)
def gpu_available() -> bool:
    """True if `nvidia-smi` is on PATH and reports at least one GPU.

    Cached (checked at most once per process): this gets consulted once per
    genome evaluation otherwise, and shelling out is not free at the
    thousand-genome scale a real evolution run evaluates at. Any failure
    (missing binary, no driver, timeout, unexpected output) means False,
    never an exception -- CPU is always the safe fallback, and a genome's
    fitness must never depend on whether GPU detection itself is flaky.
    """
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5, check=False
        )
    except Exception:
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def resolve_device(model_family: str) -> str:
    """"cuda" for an xgb genome on a machine with a usable GPU, "cpu"
    otherwise -- including every non-xgb family, which never uses a device
    at all but still gets a value here, so `FitnessResult.device` and a
    run's persisted config are never missing the field, just always "cpu"
    for models that don't distinguish.
    """
    if model_family == "xgb" and gpu_available():
        return "cuda"
    return "cpu"
