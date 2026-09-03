"""Download, cache, and calendar-align daily OHLCV data.

The S&P 500 constituent list is bundled as a static CSV (`data/tickers/sp500.csv`)
and is never re-fetched from the web at run time -- see CLAUDE.md for why
(survivorship bias is an accepted, documented limitation of step 1, not
something re-fetching the list would fix).

Prices are downloaded with `yfinance`, `auto_adjust=True`, and cached one
parquet file per ticker under `data/cache/`. Re-running does not re-download
unless `refresh=True`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_universe(path: str | Path) -> pd.DataFrame:
    """Load the bundled ticker universe (ticker, name, sector, as_of)."""
    df = pd.read_csv(path)
    expected = {"ticker", "name", "sector", "as_of"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"universe csv {path} missing columns: {missing}")
    return df


@dataclass
class DownloadReport:
    ok: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"downloaded={len(self.ok)} cached={len(self.cached)} "
            f"failed={len(self.failed)}"
        )


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{ticker}.parquet"


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path,
    refresh: bool = False,
    batch_size: int = 50,
    max_retries: int = 3,
    pause_seconds: float = 2.0,
) -> DownloadReport:
    """Download daily OHLCV for `tickers` into `cache_dir`, one parquet each.

    Batches requests to `yfinance` to reduce rate-limit pressure. A ticker
    already cached is skipped unless `refresh=True`. A ticker that fails
    after retries is recorded in the report and skipped -- one bad ticker
    does not abort the run.
    """
    import yfinance as yf

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    report = DownloadReport()
    to_fetch = []
    for t in tickers:
        if not refresh and _cache_path(cache_dir, t).exists():
            report.cached.append(t)
        else:
            to_fetch.append(t)

    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i : i + batch_size]
        attempt = 0
        data = None
        while attempt < max_retries:
            attempt += 1
            try:
                data = yf.download(
                    tickers=batch,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
                if data is not None and not data.empty:
                    break
            except Exception as e:  # noqa: BLE001 - yfinance raises many types
                logger.warning("batch download attempt %d failed: %s", attempt, e)
            time.sleep(pause_seconds * attempt)

        if data is None or data.empty:
            report.failed.extend(batch)
            continue

        for t in batch:
            try:
                if len(batch) == 1:
                    tdf = data
                else:
                    tdf = data[t] if t in data.columns.get_level_values(0) else None
                if tdf is None or tdf.empty:
                    report.failed.append(t)
                    continue
                tdf = tdf.rename(columns=str.lower)
                tdf = tdf[[c for c in REQUIRED_COLUMNS if c in tdf.columns]]
                tdf = tdf.dropna(subset=["close"])
                if tdf.empty:
                    report.failed.append(t)
                    continue
                tdf.index.name = "date"
                tdf.to_parquet(_cache_path(cache_dir, t))
                report.ok.append(t)
            except Exception as e:  # noqa: BLE001
                logger.warning("failed to cache %s: %s", t, e)
                report.failed.append(t)

        time.sleep(pause_seconds)

    return report


def _load_one(cache_dir: Path, ticker: str) -> pd.DataFrame | None:
    p = _cache_path(cache_dir, ticker)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df


@dataclass
class PanelReport:
    n_tickers_in: int = 0
    n_tickers_out: int = 0
    dropped_missing_cache: list[str] = field(default_factory=list)
    dropped_short_history: dict[str, int] = field(default_factory=dict)
    n_ticker_days_dropped_missing_close: int = 0


def load_panel(
    tickers: list[str],
    cache_dir: str | Path,
    min_history_days: int = 1000,
) -> tuple[pd.DataFrame, PanelReport]:
    """Build a long panel (date, ticker, open, high, low, close, volume).

    Aligns to the union trading calendar across all cached tickers. Drops a
    ticker-day when close is missing. Drops tickers with fewer than
    `min_history_days` remaining rows.
    """
    cache_dir = Path(cache_dir)
    report = PanelReport(n_tickers_in=len(tickers))

    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = _load_one(cache_dir, t)
        if df is None:
            report.dropped_missing_cache.append(t)
            continue
        frames[t] = df

    if not frames:
        raise RuntimeError("no cached tickers found -- run `stockml download` first")

    union_calendar = sorted(set().union(*[set(df.index) for df in frames.values()]))
    union_calendar = pd.DatetimeIndex(union_calendar)

    long_frames = []
    for t, df in frames.items():
        df = df.reindex(union_calendar)
        n_missing = df["close"].isna().sum()
        report.n_ticker_days_dropped_missing_close += int(n_missing)
        df = df.dropna(subset=["close"])
        if len(df) < min_history_days:
            report.dropped_short_history[t] = len(df)
            continue
        df = df.copy()
        df["ticker"] = t
        df.index.name = "date"
        long_frames.append(df.reset_index())

    if not long_frames:
        raise RuntimeError(
            "no tickers survived min_history_days filtering -- "
            "lower min_history_days or check the date range"
        )

    panel = pd.concat(long_frames, ignore_index=True)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    report.n_tickers_out = panel["ticker"].nunique()
    return panel, report
