"""`download_prices` must not treat "a cache file exists" as "already have
what I need" -- a file cached by an earlier, shorter-range run must be
detected as insufficient and re-fetched when a later run asks for more
history. See `data._cache_needs_refresh`.

The flip side matters just as much: a ticker whose real history starts
after the requested start (e.g. a 2012 IPO with start=2010) can never
"cover" that request, so without the `meta_path` escape hatch it would be
re-fetched from the live API on every single run -- which was observed to
actually break run-to-run reproducibility (two `evolve --quick` runs with
the same seed produced different fitness for the identical genome, because
the live API didn't return byte-identical data on two separate calls).
"""

from __future__ import annotations

import pandas as pd

from stockml.data import _cache_needs_refresh, _read_earliest_requested_start, _write_earliest_requested_start


def _write_cache(path, start: str, end: str) -> None:
    idx = pd.bdate_range(start=start, end=end)
    df = pd.DataFrame({"close": range(len(idx))}, index=idx)
    df.index.name = "date"
    df.to_parquet(path)


def test_missing_file_needs_refresh(tmp_path):
    path = tmp_path / "AAPL.parquet"
    assert _cache_needs_refresh(path, pd.Timestamp("2010-01-01")) is True


def test_short_cached_range_needs_refresh_for_earlier_request(tmp_path):
    path = tmp_path / "AAPL.parquet"
    _write_cache(path, "2024-09-01", "2026-08-31")  # e.g. cached by smoke.yaml
    assert _cache_needs_refresh(path, pd.Timestamp("2010-01-01")) is True  # step1.yaml's start


def test_cached_range_covering_request_does_not_need_refresh(tmp_path):
    path = tmp_path / "AAPL.parquet"
    _write_cache(path, "2010-01-01", "2026-08-31")
    assert _cache_needs_refresh(path, pd.Timestamp("2010-01-01")) is False
    assert _cache_needs_refresh(path, pd.Timestamp("2015-06-01")) is False  # narrower request, still covered


def test_weekend_start_within_tolerance_does_not_need_refresh(tmp_path):
    path = tmp_path / "AAPL.parquet"
    _write_cache(path, "2024-01-01", "2026-08-31")  # first trading day a few days after New Year's
    # Requesting exactly the (non-trading) first-of-year start should not
    # force a refresh just because the cache's first row is a few days later.
    assert _cache_needs_refresh(path, pd.Timestamp("2023-12-30"), tolerance_days=7) is False


def test_empty_cache_needs_refresh(tmp_path):
    path = tmp_path / "AAPL.parquet"
    pd.DataFrame({"close": []}).to_parquet(path)
    assert _cache_needs_refresh(path, pd.Timestamp("2010-01-01")) is True


def test_meta_records_earliest_requested_start(tmp_path):
    meta_path = tmp_path / "META.meta.json"
    _write_earliest_requested_start(meta_path, pd.Timestamp("2010-01-01"))
    assert _read_earliest_requested_start(meta_path) == pd.Timestamp("2010-01-01")


def test_meta_keeps_the_earliest_of_multiple_writes(tmp_path):
    meta_path = tmp_path / "META.meta.json"
    _write_earliest_requested_start(meta_path, pd.Timestamp("2015-01-01"))
    _write_earliest_requested_start(meta_path, pd.Timestamp("2010-01-01"))  # earlier -- should win
    _write_earliest_requested_start(meta_path, pd.Timestamp("2020-01-01"))  # later -- should not override
    assert _read_earliest_requested_start(meta_path) == pd.Timestamp("2010-01-01")


def test_meta_suppresses_refresh_for_a_ticker_whose_real_history_is_short(tmp_path):
    # e.g. META: IPO'd 2012, so a cache starting 2012-05-18 can never
    # "cover" a requested_start of 2010-01-01 -- without the meta escape
    # hatch this would need_refresh forever.
    path = tmp_path / "META.parquet"
    meta_path = tmp_path / "META.meta.json"
    _write_cache(path, "2012-05-18", "2026-08-31")

    # First time: no meta yet, so it looks like it needs a (re-)check.
    assert _cache_needs_refresh(path, pd.Timestamp("2010-01-01"), meta_path=meta_path) is True

    # Simulate download_prices recording that 2010-01-01 was already asked for.
    _write_earliest_requested_start(meta_path, pd.Timestamp("2010-01-01"))

    # Same (or narrower) request now: no refresh needed, no network hit.
    assert _cache_needs_refresh(path, pd.Timestamp("2010-01-01"), meta_path=meta_path) is False
    assert _cache_needs_refresh(path, pd.Timestamp("2015-01-01"), meta_path=meta_path) is False


def test_meta_does_not_suppress_refresh_for_a_genuinely_earlier_request(tmp_path):
    path = tmp_path / "META.parquet"
    meta_path = tmp_path / "META.meta.json"
    _write_cache(path, "2012-05-18", "2026-08-31")
    _write_earliest_requested_start(meta_path, pd.Timestamp("2011-01-01"))

    # Asking for something earlier than what was ever tried should still refresh.
    assert _cache_needs_refresh(path, pd.Timestamp("2005-01-01"), meta_path=meta_path) is True


def test_corrupt_meta_file_falls_back_to_needing_refresh(tmp_path):
    path = tmp_path / "META.parquet"
    meta_path = tmp_path / "META.meta.json"
    _write_cache(path, "2012-05-18", "2026-08-31")
    meta_path.write_text("not valid json", encoding="utf-8")
    assert _cache_needs_refresh(path, pd.Timestamp("2010-01-01"), meta_path=meta_path) is True
