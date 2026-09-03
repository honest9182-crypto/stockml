"""Shared test fixtures: a small deterministic synthetic OHLCV panel so most
tests don't need network access or the real price cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_synthetic_panel(
    n_tickers: int = 3, n_days: int = 300, start: str = "2020-01-01", seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    frames = []
    for i in range(n_tickers):
        ticker = f"T{i}"
        returns = rng.normal(0, 0.01, size=n_days)
        close = 100 * np.cumprod(1 + returns)
        volume = rng.integers(1_000_000, 5_000_000, size=n_days).astype(float)
        df = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "open": close * 0.999,
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "volume": volume,
            }
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    return make_synthetic_panel()
