"""evolution/device.py: `resolve_device` picks "cuda" only for the "xgb"
family, and only when `gpu_available()` says so; `gpu_available()` itself
degrades to False on any detection failure and is cached per-process.
"""

from __future__ import annotations

import pytest

from stockml.evolution import device as device_mod


@pytest.fixture(autouse=True)
def _clear_gpu_cache():
    # gpu_available is a process-wide lru_cache -- this file mocks it, so
    # every test here (and anything running after it) must see a clean cache.
    device_mod.gpu_available.cache_clear()
    yield
    device_mod.gpu_available.cache_clear()


def test_resolve_device_is_always_cpu_for_non_xgb_families(monkeypatch):
    monkeypatch.setattr(device_mod, "gpu_available", lambda: True)
    assert device_mod.resolve_device("logreg") == "cpu"
    assert device_mod.resolve_device("hgb") == "cpu"


def test_resolve_device_for_xgb_follows_gpu_availability(monkeypatch):
    monkeypatch.setattr(device_mod, "gpu_available", lambda: True)
    assert device_mod.resolve_device("xgb") == "cuda"
    monkeypatch.setattr(device_mod, "gpu_available", lambda: False)
    assert device_mod.resolve_device("xgb") == "cpu"


def test_gpu_available_is_false_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setattr(device_mod.shutil, "which", lambda name: None)
    assert device_mod.gpu_available() is False


def test_gpu_available_is_false_on_subprocess_failure(monkeypatch):
    def _raise(*args, **kwargs):
        raise OSError("nvidia-smi not runnable")

    monkeypatch.setattr(device_mod.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(device_mod.subprocess, "run", _raise)
    assert device_mod.gpu_available() is False


def test_gpu_available_is_cached_per_process(monkeypatch):
    calls = {"n": 0}

    def fake_which(name):
        calls["n"] += 1
        return None

    monkeypatch.setattr(device_mod.shutil, "which", fake_which)
    device_mod.gpu_available()
    device_mod.gpu_available()
    assert calls["n"] == 1  # second call served from cache, not re-checked
