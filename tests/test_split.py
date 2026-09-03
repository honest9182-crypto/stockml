"""Walk-forward split: strict temporal ordering, sanity slice excluded from
test, and no overlap between the three slices.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockml.split import assert_no_leakage, assign_split, compute_split_dates


def _dates(n_days: int = 500, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n_days)


def test_train_end_before_test_start():
    dates = _dates()
    split = compute_split_dates(dates, train_years=1, sanity_days=10)
    assert split.train_end < split.test_start
    assert split.test_end < split.sanity_start


def test_sanity_slice_is_exactly_the_last_n_days():
    dates = _dates()
    split = compute_split_dates(dates, train_years=1, sanity_days=10)
    sanity_dates = dates[(dates >= split.sanity_start) & (dates <= split.sanity_end)]
    assert len(sanity_dates) == 10
    assert split.sanity_end == dates[-1]


def test_assign_split_has_no_overlap_and_covers_all_dates():
    dates = _dates()
    split = compute_split_dates(dates, train_years=1, sanity_days=10)
    df = pd.DataFrame({"date": dates})
    df["split"] = assign_split(df["date"], split)

    assert not (df["split"] == "unassigned").any()
    assert_no_leakage(df, "split", "date")

    train_dates = set(df.loc[df["split"] == "train", "date"])
    test_dates = set(df.loc[df["split"] == "test", "date"])
    sanity_dates = set(df.loc[df["split"] == "sanity", "date"])
    assert train_dates.isdisjoint(test_dates)
    assert train_dates.isdisjoint(sanity_dates)
    assert test_dates.isdisjoint(sanity_dates)
    assert train_dates | test_dates | sanity_dates == set(dates)


def test_sanity_days_too_large_raises():
    dates = _dates(n_days=5)
    with pytest.raises(ValueError):
        compute_split_dates(dates, train_years=1, sanity_days=10)
