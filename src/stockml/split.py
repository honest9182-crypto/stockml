"""Walk-forward, time-based train/test/sanity splitting.

- `train_years` from the start of the data forms the training set.
- Every subsequent day is test, predicted in chronological order.
- The final `sanity_days` trading days are reserved as their own slice --
  reported on its own line, never merged into test metrics, never used to
  decide anything. It exists only to prove the pipeline works on the
  freshest data.
- A one-day embargo (`embargo_days`, default 1) is cut between train and
  test: the last `embargo_days` day(s) that would otherwise be the final
  training day(s) are dropped entirely -- not train, not test, not sanity.
  Reason: the training row for day `t` carries `label(t)`, and
  `label(t)` is computed from `r_next(t) = close(t+1)/close(t) - 1` (see
  labels.py). If `t` is the last training day, `t+1` is the *first test
  day* -- so training on that row means the label the model is fit on
  directly encodes the first test day's close. That's not a feature
  leaking test information, it's the *label* doing it, which no amount of
  feature-side care (shift tests, truncation tests) would ever catch. The
  embargo removes the row before it can happen. (The same asymmetry exists
  one day earlier, between the last test day and the first sanity day; it
  is deliberately left alone so the sanity slice stays exactly "the final
  `sanity_days` trading days" as specified.)

Pooled training across all tickers is the default; `per_ticker` only changes
how models are fit in `run.py`, not how dates are split here.

This module asserts strict temporal ordering at runtime (not just in tests):
no train timestamp >= any test timestamp, and train/test/sanity never overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitDates:
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # inclusive: last date in train
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # inclusive: last date in test (before sanity)
    sanity_start: pd.Timestamp
    sanity_end: pd.Timestamp  # inclusive: last date overall


def compute_split_dates(
    unique_dates: pd.DatetimeIndex,
    train_years: float,
    sanity_days: int,
    embargo_days: int = 1,
) -> SplitDates:
    """Compute the train/test/sanity date boundaries from the set of dates present.

    `train_years` marks off a calendar window from the first date; every
    trading day strictly after that window and before the sanity slice is
    test. The last `sanity_days` trading days (by count, not calendar time)
    are the sanity slice. The last `embargo_days` day(s) that would
    otherwise be the final training day(s) are dropped instead -- see the
    module docstring for why.
    """
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(unique_dates).unique()))
    if len(dates) == 0:
        raise ValueError("no dates to split")
    if sanity_days < 1 or sanity_days >= len(dates):
        raise ValueError(
            f"sanity_days={sanity_days} invalid for {len(dates)} available dates"
        )
    if embargo_days < 0:
        raise ValueError(f"embargo_days={embargo_days} must be >= 0")

    train_start = dates[0]
    train_cutoff = train_start + pd.DateOffset(years=train_years)

    sanity_dates = dates[-sanity_days:]
    sanity_start, sanity_end = sanity_dates[0], sanity_dates[-1]

    pre_sanity = dates[dates < sanity_start]
    train_dates = pre_sanity[pre_sanity < train_cutoff]
    test_dates = pre_sanity[pre_sanity >= train_cutoff]

    if len(train_dates) <= embargo_days:
        raise ValueError(
            f"train_years={train_years} leaves only {len(train_dates)} training "
            f"day(s), not enough to also drop the {embargo_days}-day embargo -- "
            f"check the date range in the config"
        )
    if len(test_dates) == 0:
        raise ValueError(
            "zero test days between the end of training and the sanity slice -- "
            "widen the date range or shrink train_years/sanity_days"
        )

    train_dates = train_dates[: len(train_dates) - embargo_days] if embargo_days else train_dates
    train_end = train_dates[-1]
    test_start, test_end = test_dates[0], test_dates[-1]

    split = SplitDates(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        sanity_start=sanity_start,
        sanity_end=sanity_end,
    )
    _assert_ordering(split)
    return split


def _assert_ordering(split: SplitDates) -> None:
    assert split.train_end < split.test_start, (
        f"train_end {split.train_end} >= test_start {split.test_start}"
    )
    assert split.test_end < split.sanity_start, (
        f"test_end {split.test_end} >= sanity_start {split.sanity_start}"
    )
    assert split.train_start <= split.train_end
    assert split.test_start <= split.test_end
    assert split.sanity_start <= split.sanity_end


def assign_split(dates: pd.Series, split: SplitDates) -> pd.Series:
    """Label each row 'train' / 'test' / 'sanity' by its date, per `split`."""
    dates = pd.to_datetime(dates)
    out = pd.Series("unassigned", index=dates.index, dtype=object)
    out[(dates >= split.train_start) & (dates <= split.train_end)] = "train"
    out[(dates >= split.test_start) & (dates <= split.test_end)] = "test"
    out[(dates >= split.sanity_start) & (dates <= split.sanity_end)] = "sanity"
    return out


def assert_no_leakage(df: pd.DataFrame, split_col: str, date_col: str) -> None:
    """Runtime guard: no train date >= any test date, no overlap between slices."""
    train_dates = df.loc[df[split_col] == "train", date_col]
    test_dates = df.loc[df[split_col] == "test", date_col]
    sanity_dates = df.loc[df[split_col] == "sanity", date_col]
    if len(train_dates) and len(test_dates):
        assert train_dates.max() < test_dates.min(), "train/test date overlap detected"
    if len(test_dates) and len(sanity_dates):
        assert test_dates.max() < sanity_dates.min(), "test/sanity date overlap detected"
    if len(train_dates) and len(sanity_dates):
        assert train_dates.max() < sanity_dates.min(), "train/sanity date overlap detected"
