"""Three hard-walled, non-overlapping time zones for the evolutionary search
(CLAUDE.md's "Evolutionary search" section):

- train: the existing step-1 training window (start of data -> the
  train/test embargo). Every genome fits here.
- arena: test days from the start of test through a config-fixed
  `arena_end` date. Fitness is measured here, thousands of times over the
  course a run -- evolution *will* overfit to it, which is exactly why the
  third zone exists.
- vault: a config-fixed `vault_start` date through the start of the sanity
  slice. Never touched during evolution; opened once, for a fixed,
  pre-declared list of individuals, by `evolution/vault.py` (which enforces
  this at runtime, not just by convention -- it raises if called while an
  evolution is in progress).

`arena_end` and `vault_start` are adjacent calendar dates (contiguous zones,
no gap) so the arena's last day's label technically depends on the vault's
first day's close (same asymmetry as the step-1 train/test embargo). This is
deliberately *not* embargoed here, for the same reason step 1 never
embargoed test/sanity: a zone's label being *evaluated* against doesn't leak
anything into a fit the way a zone's label being *trained on* would --
nothing in the vault is ever read by anything that shapes a genome's fitness
or a model's parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockml.split import SplitDates, compute_split_dates


@dataclass(frozen=True)
class EvoZones:
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # embargoed, same as split.SplitDates.train_end
    arena_start: pd.Timestamp
    arena_end: pd.Timestamp
    vault_start: pd.Timestamp
    vault_end: pd.Timestamp
    sanity_start: pd.Timestamp
    sanity_end: pd.Timestamp


def compute_evo_zones(
    unique_dates: pd.DatetimeIndex,
    train_years: float,
    sanity_days: int,
    embargo_days: int,
    arena_end: str | pd.Timestamp,
    vault_start: str | pd.Timestamp,
) -> EvoZones:
    """Reuses `split.compute_split_dates` for train/embargo/sanity, then
    splits its "test" zone in two at the config-fixed arena/vault boundary.
    """
    split: SplitDates = compute_split_dates(unique_dates, train_years, sanity_days, embargo_days)
    arena_end = pd.Timestamp(arena_end)
    vault_start = pd.Timestamp(vault_start)

    if arena_end >= vault_start:
        raise ValueError(f"arena_end {arena_end} must be before vault_start {vault_start}")

    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(unique_dates).unique()))
    test_dates = dates[(dates >= split.test_start) & (dates <= split.test_end)]

    arena_dates = test_dates[test_dates <= arena_end]
    vault_dates = test_dates[test_dates >= vault_start]

    if len(arena_dates) == 0:
        raise ValueError(
            f"zero arena days: test starts {split.test_start}, arena_end is {arena_end}"
        )
    if len(vault_dates) == 0:
        raise ValueError(
            f"zero vault days: test ends {split.test_end}, vault_start is {vault_start}"
        )

    zones = EvoZones(
        train_start=split.train_start,
        train_end=split.train_end,
        arena_start=arena_dates[0],
        arena_end=arena_dates[-1],
        vault_start=vault_dates[0],
        vault_end=vault_dates[-1],
        sanity_start=split.sanity_start,
        sanity_end=split.sanity_end,
    )
    _assert_zone_ordering(zones)
    return zones


def _assert_zone_ordering(zones: EvoZones) -> None:
    assert zones.train_end < zones.arena_start, "train/arena overlap"
    assert zones.arena_end < zones.vault_start, "arena/vault overlap"
    assert zones.vault_end < zones.sanity_start, "vault/sanity overlap"
    assert zones.train_start <= zones.train_end
    assert zones.arena_start <= zones.arena_end
    assert zones.vault_start <= zones.vault_end
    assert zones.sanity_start <= zones.sanity_end


def assign_zone(dates: pd.Series, zones: EvoZones) -> pd.Series:
    """Label each row 'train' / 'arena' / 'vault' / 'sanity' by its date."""
    dates = pd.to_datetime(dates)
    out = pd.Series("unassigned", index=dates.index, dtype=object)
    out[(dates >= zones.train_start) & (dates <= zones.train_end)] = "train"
    out[(dates >= zones.arena_start) & (dates <= zones.arena_end)] = "arena"
    out[(dates >= zones.vault_start) & (dates <= zones.vault_end)] = "vault"
    out[(dates >= zones.sanity_start) & (dates <= zones.sanity_end)] = "sanity"
    return out
