"""Past-only per-ticker features.

Every feature at row `t` is built exclusively from data with timestamp `<= t`:
lags, rolling windows (which include day `t` itself -- legitimate, since a
day's own close/volume are known by the time that day's row is used), and the
day's own calendar date. Nothing here ever reads `t+1`.

Features are computed per ticker (`build_features` takes a single ticker's
frame) so the design stays open to cross-sectional/market-wide features later
without reshaping anything -- see CLAUDE.md.

`sigma`/`band` reuse `labels.rolling_sigma`, the *same* function the label
builder uses, so the volatility band the label was cut with and the band the
model is shown as a feature can never silently diverge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockml.labels import daily_returns, rolling_sigma

RETURN_LAGS = [1, 2, 3, 5, 10]
MEAN_STD_WINDOWS = [5, 20]
SMA_WINDOWS = [20, 60]
VOLUME_Z_WINDOW = 20
DAY_NAMES = ["mon", "tue", "wed", "thu", "fri"]  # trading days only


def feature_names() -> list[str]:
    """Column names `build_features` produces, in a fixed order."""
    names = [f"log_ret_lag_{n}" for n in RETURN_LAGS]
    names += [f"ret_mean_{w}" for w in MEAN_STD_WINDOWS]
    names += [f"ret_std_{w}" for w in MEAN_STD_WINDOWS]
    names += ["ret_std_ratio_5_20"]
    names += ["close_sma20_ratio", "close_sma60_ratio", "sma20_sma60_ratio"]
    names += ["volume_zscore_20"]
    names += [f"dow_{d}" for d in DAY_NAMES[1:]]  # drop_first -> mon is baseline
    names += ["sigma", "band"]
    return names


def build_features(
    df: pd.DataFrame, k: float, vol_window: int
) -> pd.DataFrame:
    """Build the feature set for a single ticker's OHLCV frame.

    `df` must be sorted by date ascending with columns date, close, volume.
    Returns `df` with feature columns appended (input columns preserved).
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    close = df["close"]
    volume = df["volume"]

    log_close = np.log(close)
    for n in RETURN_LAGS:
        df[f"log_ret_lag_{n}"] = log_close - log_close.shift(n)

    ret = daily_returns(close)
    for w in MEAN_STD_WINDOWS:
        df[f"ret_mean_{w}"] = ret.rolling(window=w, min_periods=w).mean()
        df[f"ret_std_{w}"] = ret.rolling(window=w, min_periods=w).std(ddof=1)
    df["ret_std_ratio_5_20"] = df["ret_std_5"] / df["ret_std_20"]

    sma = {w: close.rolling(window=w, min_periods=w).mean() for w in SMA_WINDOWS}
    df["close_sma20_ratio"] = close / sma[20] - 1
    df["close_sma60_ratio"] = close / sma[60] - 1
    df["sma20_sma60_ratio"] = sma[20] / sma[60] - 1

    vol_mean = volume.rolling(window=VOLUME_Z_WINDOW, min_periods=VOLUME_Z_WINDOW).mean()
    vol_std = volume.rolling(window=VOLUME_Z_WINDOW, min_periods=VOLUME_Z_WINDOW).std(ddof=1)
    df["volume_zscore_20"] = (volume - vol_mean) / vol_std

    dow = pd.to_datetime(df["date"]).dt.dayofweek  # 0=Mon .. 4=Fri (trading days)
    dow_dummies = pd.get_dummies(dow, prefix="dow")
    rename = {i: f"dow_{DAY_NAMES[i]}" for i in range(5)}
    dow_dummies = dow_dummies.rename(columns=rename)
    for d in DAY_NAMES[1:]:  # drop_first: mon is the baseline
        col = f"dow_{d}"
        df[col] = dow_dummies[col].astype(float) if col in dow_dummies else 0.0

    df["sigma"] = rolling_sigma(ret, vol_window)
    df["band"] = k * df["sigma"]

    return df


def build_features_panel(panel: pd.DataFrame, k: float, vol_window: int) -> pd.DataFrame:
    """Apply `build_features` to every ticker in a long panel."""
    parts = []
    for ticker, g in panel.groupby("ticker", sort=False):
        parts.append(build_features(g, k=k, vol_window=vol_window))
    return pd.concat(parts, ignore_index=True)
