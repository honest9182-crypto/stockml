"""Three-class (up / down / stagnant) next-day direction labels.

For each ticker and day `t`:

    r_next(t) = close(t+1) / close(t) - 1
    sigma(t)  = std of daily returns over the trailing `vol_window` days
                ending at t (uses days <= t only)
    band(t)   = k * sigma(t)

    label(t) = up        if r_next(t) >  band(t)
             = down      if r_next(t) < -band(t)
             = stagnant  otherwise

`sigma`/`band` are computed here once and imported by `features.py`, so the
band the label uses and the band the model sees can never silently diverge.

The last row of each ticker has no label (no t+1 to look at) and is dropped,
never filled. Rows without a full `vol_window` of trailing history also have
no `sigma` and are dropped for the same reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ["down", "stagnant", "up"]


def daily_returns(close: pd.Series) -> pd.Series:
    """Simple daily return r(t) = close(t)/close(t-1) - 1. Uses data <= t only."""
    return close / close.shift(1) - 1


def rolling_sigma(returns: pd.Series, window: int) -> pd.Series:
    """Trailing std of `returns` over `window` days ending at t (data <= t only).

    Requires a full window of observations (min_periods=window); earlier rows
    are NaN rather than computed from a partial, inconsistent window.
    """
    return returns.rolling(window=window, min_periods=window).std(ddof=1)


def compute_labels_one_ticker(df: pd.DataFrame, k: float, vol_window: int) -> pd.DataFrame:
    """Add sigma, band, r_next, label columns to a single ticker's OHLCV frame.

    `df` must be sorted by date ascending and contain a `close` column.
    Rows where the label cannot be computed (no t+1, or insufficient
    trailing history for sigma) get `label = NaN` -- the caller drops them.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    ret = daily_returns(df["close"])
    df["sigma"] = rolling_sigma(ret, vol_window)
    df["band"] = k * df["sigma"]
    df["r_next"] = df["close"].shift(-1) / df["close"] - 1

    label = np.full(len(df), "stagnant", dtype=object)
    label[(df["r_next"] > df["band"]).to_numpy()] = "up"
    label[(df["r_next"] < -df["band"]).to_numpy()] = "down"
    df["label"] = label

    invalid = df["r_next"].isna() | df["sigma"].isna()
    df.loc[invalid, "label"] = np.nan
    return df


def build_labels(panel: pd.DataFrame, k: float, vol_window: int) -> pd.DataFrame:
    """Compute labels for every ticker in a long panel and drop invalid rows.

    `panel` must have columns date, ticker, close (and typically open/high/low/volume,
    which pass through untouched).
    """
    parts = []
    for ticker, g in panel.groupby("ticker", sort=False):
        parts.append(compute_labels_one_ticker(g, k=k, vol_window=vol_window))
    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["label"]).reset_index(drop=True)
    return out


def class_distribution(labels: pd.Series) -> dict[str, float]:
    """Fraction of rows in each class, always including all three keys."""
    counts = labels.value_counts()
    total = len(labels)
    return {c: (counts.get(c, 0) / total if total else 0.0) for c in CLASSES}


def check_class_balance(
    dist: dict[str, float], low: float = 0.15, high: float = 0.60
) -> list[str]:
    """Return warning strings for any class outside [low, high] share."""
    warnings = []
    for cls, frac in dist.items():
        if frac < low:
            warnings.append(
                f"class '{cls}' is {frac:.1%} of rows (< {low:.0%}) -- "
                f"consider adjusting k before trusting accuracy"
            )
        elif frac > high:
            warnings.append(
                f"class '{cls}' is {frac:.1%} of rows (> {high:.0%}) -- "
                f"consider adjusting k before trusting accuracy"
            )
    return warnings
