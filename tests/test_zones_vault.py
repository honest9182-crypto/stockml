"""Time zones: arena and vault never overlap, both exclude train and
sanity. The vault loader raises while an evolution is recorded as running.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from stockml.evolution.zones import assign_zone, compute_evo_zones
from stockml.evolution.vault import VaultGuardError, _assert_not_running


def _dates(n_days: int = 4000, start: str = "2010-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n_days)


def test_zones_never_overlap_and_exclude_train_and_sanity():
    dates = _dates()
    zones = compute_evo_zones(
        dates, train_years=3, sanity_days=10, embargo_days=1,
        arena_end="2019-12-31", vault_start="2020-01-01",
    )
    assert zones.train_end < zones.arena_start
    assert zones.arena_end < zones.vault_start
    assert zones.vault_end < zones.sanity_start

    df = pd.DataFrame({"date": dates})
    df["zone"] = assign_zone(df["date"], zones)

    train = set(df.loc[df["zone"] == "train", "date"])
    arena = set(df.loc[df["zone"] == "arena", "date"])
    vault = set(df.loc[df["zone"] == "vault", "date"])
    sanity = set(df.loc[df["zone"] == "sanity", "date"])

    assert train.isdisjoint(arena)
    assert train.isdisjoint(vault)
    assert train.isdisjoint(sanity)
    assert arena.isdisjoint(vault)
    assert arena.isdisjoint(sanity)
    assert vault.isdisjoint(sanity)
    assert len(arena) > 0 and len(vault) > 0


def test_arena_end_before_vault_start_enforced():
    dates = _dates()
    with pytest.raises(ValueError):
        compute_evo_zones(
            dates, train_years=3, sanity_days=10, embargo_days=1,
            arena_end="2020-01-01", vault_start="2019-12-31",  # backwards
        )


def test_zero_arena_days_raises():
    dates = _dates()
    with pytest.raises(ValueError):
        compute_evo_zones(
            dates, train_years=3, sanity_days=10, embargo_days=1,
            arena_end="2010-01-01", vault_start="2010-01-02",  # before test even starts
        )


def test_vault_guard_raises_while_running(tmp_path):
    run_dir = tmp_path / "evo_run"
    run_dir.mkdir()
    with open(run_dir / "progress.json", "w", encoding="utf-8") as f:
        json.dump({"status": "running", "last_completed_generation": 2}, f)
    with pytest.raises(VaultGuardError):
        _assert_not_running(run_dir)


def test_vault_guard_allows_completed(tmp_path):
    run_dir = tmp_path / "evo_run"
    run_dir.mkdir()
    with open(run_dir / "progress.json", "w", encoding="utf-8") as f:
        json.dump({"status": "completed", "last_completed_generation": 24}, f)
    _assert_not_running(run_dir)  # must not raise


def test_vault_guard_allows_halted_leak_alarm(tmp_path):
    run_dir = tmp_path / "evo_run"
    run_dir.mkdir()
    with open(run_dir / "progress.json", "w", encoding="utf-8") as f:
        json.dump({"status": "halted_leak_alarm", "last_completed_generation": 3}, f)
    _assert_not_running(run_dir)  # not "running" -- must not raise
