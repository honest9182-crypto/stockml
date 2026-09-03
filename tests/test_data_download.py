"""`download_prices` batch-download parsing: yfinance returns a
(ticker, field) MultiIndex-columned frame for `group_by="ticker"` requests
regardless of batch size -- including a batch of exactly one ticker -- so a
single-ticker batch must be unwrapped the same way a multi-ticker one is.
(A real single-ticker download once silently produced zero usable rows
because of this.)
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockml.data import download_prices


def _multiindex_frame(ticker: str, n: int = 5) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    fields = ["Open", "High", "Low", "Close", "Volume"]
    cols = pd.MultiIndex.from_product([[ticker], fields], names=["Ticker", "Price"])
    data = {(ticker, f): range(n) for f in fields}
    df = pd.DataFrame(data, index=dates, columns=cols)
    df.index.name = "Date"
    return df


def test_single_ticker_batch_unwraps_multiindex(tmp_path, monkeypatch):
    calls = []

    def fake_download(tickers, **kwargs):
        calls.append(list(tickers))
        return _multiindex_frame(tickers[0])

    import yfinance

    monkeypatch.setattr(yfinance, "download", fake_download)

    report = download_prices(["SOLO"], "2020-01-01", "2020-01-10", tmp_path, batch_size=50)

    assert report.ok == ["SOLO"]
    assert report.failed == []
    cached = pd.read_parquet(tmp_path / "SOLO.parquet")
    assert not cached.empty
    assert "close" in cached.columns
    assert len(calls) == 1  # one batch call, no wasted retries


def test_multi_ticker_batch_still_works(tmp_path, monkeypatch):
    def fake_download(tickers, **kwargs):
        frames = [_multiindex_frame(t) for t in tickers]
        return pd.concat(frames, axis=1)

    import yfinance

    monkeypatch.setattr(yfinance, "download", fake_download)

    report = download_prices(["AAA", "BBB"], "2020-01-01", "2020-01-10", tmp_path, batch_size=50)

    assert set(report.ok) == {"AAA", "BBB"}
    for t in ["AAA", "BBB"]:
        cached = pd.read_parquet(tmp_path / f"{t}.parquet")
        assert "close" in cached.columns
