"""Walk-forward split: strict temporal ordering, sanity slice excluded from
test, no overlap between the three slices, and the train/test embargo.
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


def test_assign_split_has_no_overlap_and_covers_all_dates_except_the_embargo():
    dates = _dates()
    split = compute_split_dates(dates, train_years=1, sanity_days=10, embargo_days=1)
    df = pd.DataFrame({"date": dates})
    df["split"] = assign_split(df["date"], split)

    assert_no_leakage(df, "split", "date")

    train_dates = set(df.loc[df["split"] == "train", "date"])
    test_dates = set(df.loc[df["split"] == "test", "date"])
    sanity_dates = set(df.loc[df["split"] == "sanity", "date"])
    assert train_dates.isdisjoint(test_dates)
    assert train_dates.isdisjoint(sanity_dates)
    assert test_dates.isdisjoint(sanity_dates)

    # Exactly the embargoed day(s) are unassigned -- not train, not test, not sanity.
    unassigned = set(df.loc[df["split"] == "unassigned", "date"])
    assert len(unassigned) == 1
    assert train_dates | test_dates | sanity_dates | unassigned == set(dates)


def test_embargo_drops_the_last_training_day():
    dates = _dates()
    no_embargo = compute_split_dates(dates, train_years=1, sanity_days=10, embargo_days=0)
    embargoed = compute_split_dates(dates, train_years=1, sanity_days=10, embargo_days=1)

    # Same test/sanity boundaries either way -- only the train tail moves.
    assert embargoed.test_start == no_embargo.test_start
    assert embargoed.sanity_start == no_embargo.sanity_start
    assert embargoed.train_end < no_embargo.train_end

    # The embargoed day is exactly the day that used to be the last training day,
    # and it's the day whose label would be r_next(test_start's close).
    dropped_day = no_embargo.train_end
    assert dropped_day > embargoed.train_end
    assert dropped_day < embargoed.test_start


def test_sanity_days_too_large_raises():
    dates = _dates(n_days=5)
    with pytest.raises(ValueError):
        compute_split_dates(dates, train_years=1, sanity_days=10)


def test_embargo_too_large_raises():
    dates = _dates(n_days=5)
    with pytest.raises(ValueError):
        compute_split_dates(dates, train_years=1, sanity_days=1, embargo_days=10)
