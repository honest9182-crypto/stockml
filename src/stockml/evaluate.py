"""Metrics, per-ticker/per-class breakdowns, significance testing, and the
leak diagnostics that run automatically when the leak alarm fires
(CLAUDE.md principle 3: a high score is a bug until proven otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from stockml.models.base import CLASS_ORDER

# ---------------------------------------------------------------------------
# Core classification metrics
# ---------------------------------------------------------------------------


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    """Accuracy, balanced accuracy, macro F1, per-class P/R/support, confusion matrix."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0
    )
    per_class = {
        cls: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(CLASS_ORDER)
    }
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "n": int(len(y_true)),
        "per_class": per_class,
        "confusion_matrix": {"labels": CLASS_ORDER, "matrix": cm.tolist()},
    }


# ---------------------------------------------------------------------------
# Per-ticker breakdown
# ---------------------------------------------------------------------------


def per_ticker_accuracy(ticker: pd.Series, y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Accuracy per ticker, indexed by ticker."""
    correct = (y_true.to_numpy() == y_pred.to_numpy()).astype(float)
    return pd.Series(correct, index=ticker.index).groupby(ticker).mean()


def per_ticker_summary(
    ticker: pd.Series, y_true: pd.Series, y_pred_model: pd.Series, y_pred_baseline: pd.Series
) -> dict[str, Any]:
    """Mean/median/p10/p90 ticker accuracy, and count of tickers beating the baseline."""
    model_acc = per_ticker_accuracy(ticker, y_true, y_pred_model)
    baseline_acc = per_ticker_accuracy(ticker, y_true, y_pred_baseline)
    n_beating = int((model_acc.reindex(baseline_acc.index) > baseline_acc).sum())
    return {
        "n_tickers": int(len(model_acc)),
        "mean": float(model_acc.mean()),
        "median": float(model_acc.median()),
        "p10": float(model_acc.quantile(0.10)),
        "p90": float(model_acc.quantile(0.90)),
        "n_beating_majority_baseline": n_beating,
    }


# ---------------------------------------------------------------------------
# Daily / rolling accuracy
# ---------------------------------------------------------------------------


def daily_accuracy(date: pd.Series, y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Pooled-across-tickers accuracy per calendar date, sorted by date.

    This is the unit of analysis for the day-level significance test below:
    one number per trading day, not one per (ticker, day) row.
    """
    correct = (y_true.to_numpy() == y_pred.to_numpy()).astype(float)
    return pd.Series(correct, index=pd.to_datetime(date)).groupby(level=0).mean().sort_index()


def rolling_accuracy_series(
    date: pd.Series, y_true: pd.Series, y_pred: pd.Series, window: int = 60
) -> pd.Series:
    """Pooled-across-tickers accuracy per date, then a rolling mean over `window`
    trading days -- so a lucky stretch shows up visibly as a stretch, not a headline number.
    """
    daily = daily_accuracy(date, y_true, y_pred)
    return daily.rolling(window=window, min_periods=max(1, window // 3)).mean()


# ---------------------------------------------------------------------------
# Prediction mix
# ---------------------------------------------------------------------------


def prediction_mix(y_pred: pd.Series) -> dict[str, float]:
    """Share of each class the model actually predicted (not the true labels).
    Surfaces a model that, say, just predicts 'stagnant' every time.
    """
    counts = y_pred.value_counts()
    total = len(y_pred)
    return {c: (counts.get(c, 0) / total if total else 0.0) for c in CLASS_ORDER}


def save_rolling_accuracy_plot(series: pd.Series, path: str | Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.to_numpy())
    ax.axhline(1 / 3, color="grey", linestyle="--", linewidth=1, label="1/3 (random)")
    ax.set_title(title)
    ax.set_ylabel("rolling accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Significance vs. baseline
# ---------------------------------------------------------------------------


def binomial_test_vs_baseline(n_hits: int, n_trials: int, baseline_rate: float) -> dict[str, Any]:
    """One-sided binomial test: is the model's hit rate > the baseline rate?

    NOTE: this treats every (ticker, day) row as an independent Bernoulli
    trial. It isn't -- on a single day, hundreds of tickers move together on
    market-wide news, so this massively overstates the effective sample size
    and therefore overstates significance. Kept in metrics.json labelled
    "row_level_overstated" for comparison; not shown in report.txt.
    `day_level_paired_test` below is the honest version. See CLAUDE.md.
    """
    result = binomtest(n_hits, n_trials, baseline_rate, alternative="greater")
    return {
        "n_hits": int(n_hits),
        "n_trials": int(n_trials),
        "hit_rate": float(n_hits / n_trials) if n_trials else 0.0,
        "baseline_rate": float(baseline_rate),
        "p_value": float(result.pvalue),
    }


def block_bootstrap_ci(
    x: np.ndarray, block_size: int = 20, n_boot: int = 2000, seed: int = 0, ci: float = 0.95
) -> tuple[float, float]:
    """Moving block bootstrap CI for the mean of a daily series `x`.

    Resamples overlapping blocks of `block_size` consecutive days (with
    replacement) rather than individual days, so day-to-day autocorrelation
    in the edge series is preserved in the resampling -- a plain i.i.d.
    bootstrap over days would understate the CI width for the same reason
    the row-level binomial test overstates significance.
    """
    n = len(x)
    if n == 0 or n_boot <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    max_start = max(n - block_size, 0)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([x[s : s + block_size] for s in starts])[:n]
        boot_means[b] = sample.mean()
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def summarize_daily_series(
    edge: pd.Series, block_size: int = 20, n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """The shared "one number per trading day" significance summary: mean
    edge (percentage points), its standard error, a one-sided paired t-test
    against zero, and a 20-day block-bootstrap 95% CI. `edge` is whatever
    daily series the caller has already reduced its comparison to -- model
    accuracy minus baseline accuracy for `day_level_paired_test` below, or
    the picker's precision-minus-base-rate edge (`picker/evaluate.py`).
    Both are "one number per trading day, not per row" (CLAUDE.md's
    day-level-vs-row-level section) and share this exact treatment so they
    can never quietly diverge in how they're judged.
    """
    edge = edge.dropna()
    if len(edge) < 2:
        return {
            "n_days": int(len(edge)),
            "mean_edge_pp": float(edge.mean() * 100) if len(edge) else 0.0,
            "se_pp": 0.0,
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "ci95_low_pp": float("nan"),
            "ci95_high_pp": float("nan"),
            "block_size": block_size,
        }

    edge_arr = edge.to_numpy()
    n = len(edge_arr)
    se_pp = float(edge_arr.std(ddof=1) / np.sqrt(n)) * 100
    t_stat, p_value = ttest_1samp(edge_arr, popmean=0.0, alternative="greater")
    ci_low, ci_high = block_bootstrap_ci(edge_arr, block_size=block_size, n_boot=n_boot, seed=seed)
    return {
        "n_days": int(n),
        "mean_edge_pp": float(edge.mean() * 100),
        "se_pp": se_pp,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ci95_low_pp": float(ci_low * 100),
        "ci95_high_pp": float(ci_high * 100),
        "block_size": block_size,
    }


def day_level_paired_test(
    date: pd.Series,
    y_true: pd.Series,
    y_pred_model: pd.Series,
    y_pred_baseline: pd.Series,
    block_size: int = 20,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """The honest significance test: pair the model against the baseline at
    the level of independent trading days, not individual (ticker, day)
    rows (see `binomial_test_vs_baseline`'s docstring for why the row-level
    version overstates significance).

    For each trading day, computes (model's pooled accuracy that day) minus
    (baseline's pooled accuracy that day) -- one number per day -- and
    hands it to `summarize_daily_series` for the mean/t-test/CI.
    """
    model_daily = daily_accuracy(date, y_true, y_pred_model)
    baseline_daily = daily_accuracy(date, y_true, y_pred_baseline)
    edge = (model_daily - baseline_daily).dropna()
    return summarize_daily_series(edge, block_size=block_size, n_boot=n_boot, seed=seed)


def yearly_edge_table(
    date: pd.Series, y_true: pd.Series, y_pred_model: pd.Series, y_pred_baseline: pd.Series
) -> dict[int, float]:
    """Mean daily edge (model accuracy - baseline accuracy), in percentage
    points, grouped by calendar year -- so a result that's really just one
    good (or bad) year is visible as one good (or bad) year.
    """
    model_daily = daily_accuracy(date, y_true, y_pred_model)
    baseline_daily = daily_accuracy(date, y_true, y_pred_baseline)
    edge = (model_daily - baseline_daily).dropna()
    by_year = edge.groupby(edge.index.year).mean() * 100
    return {int(y): float(v) for y, v in by_year.items()}


# ---------------------------------------------------------------------------
# Leak alarm
# ---------------------------------------------------------------------------


def check_leak_alarm(accuracy: float, n_test_days: int, threshold: float, min_days: int = 250) -> bool:
    """True if accuracy exceeds `threshold` on more than `min_days` test days.
    A high score is treated as a bug until leak diagnostics clear it.
    """
    return accuracy > threshold and n_test_days > min_days


# ---------------------------------------------------------------------------
# Leak diagnostics
# ---------------------------------------------------------------------------


def shift_test(
    df: pd.DataFrame,
    model: Any,
    feature_cols: list[str],
    ticker_col: str = "ticker",
    label_col: str = "label",
) -> dict[str, Any]:
    """Shift every feature one extra day staler (row t gets t-1's features) and
    re-evaluate the *already-fitted* model. A small accuracy drop is normal
    (features are staler); a collapse toward the baseline means the original
    features were carrying information from t+1.
    """
    from stockml.models.base import predict_labels

    shifted = df.copy()
    shifted[feature_cols] = shifted.groupby(ticker_col)[feature_cols].shift(1)
    shifted = shifted.dropna(subset=feature_cols)

    y_true = shifted[label_col]
    y_pred = predict_labels(model, shifted[feature_cols])
    shifted_acc = float(accuracy_score(y_true, y_pred))

    orig_y_pred = predict_labels(model, df[feature_cols])
    orig_acc = float(accuracy_score(df[label_col], orig_y_pred))

    return {
        "original_accuracy": orig_acc,
        "shifted_accuracy": shifted_acc,
        "drop": orig_acc - shifted_acc,
        "n_rows": int(len(shifted)),
    }


def label_alignment_audit(
    labeled_panel: pd.DataFrame,
    cache_dir: str | Path,
    k: float,
    vol_window: int,
    n: int = 50,
    seed: int = 0,
) -> dict[str, Any]:
    """Recompute r_next/label for `n` random (ticker, day) rows straight from
    the raw cached parquet files (bypassing the panel-building pipeline) and
    assert they match. Catches alignment/off-by-one bugs the pipeline's own
    formulas wouldn't catch (since it would just agree with itself).
    """
    cache_dir = Path(cache_dir)
    rng = np.random.default_rng(seed)
    n = min(n, len(labeled_panel))
    sample = labeled_panel.sample(n=n, random_state=seed)

    mismatches = []
    for _, row in sample.iterrows():
        ticker, date = row["ticker"], pd.Timestamp(row["date"])
        raw = pd.read_parquet(cache_dir / f"{ticker}.parquet")
        raw.index = pd.to_datetime(raw.index)
        raw = raw.sort_index()

        if date not in raw.index:
            mismatches.append({"ticker": ticker, "date": str(date), "reason": "date not in raw cache"})
            continue
        loc = raw.index.get_loc(date)
        if loc + 1 >= len(raw):
            mismatches.append({"ticker": ticker, "date": str(date), "reason": "no next day in raw cache"})
            continue

        close_t = raw["close"].iloc[loc]
        close_next = raw["close"].iloc[loc + 1]
        r_next_hand = close_next / close_t - 1

        returns_hist = raw["close"].iloc[: loc + 1].pct_change().dropna()
        if len(returns_hist) < vol_window:
            mismatches.append({"ticker": ticker, "date": str(date), "reason": "insufficient history for sigma"})
            continue
        sigma_hand = returns_hist.iloc[-vol_window:].std(ddof=1)
        band_hand = k * sigma_hand
        if r_next_hand > band_hand:
            label_hand = "up"
        elif r_next_hand < -band_hand:
            label_hand = "down"
        else:
            label_hand = "stagnant"

        ok = (
            np.isclose(r_next_hand, row["r_next"], atol=1e-9)
            and np.isclose(sigma_hand, row["sigma"], atol=1e-9)
            and label_hand == row["label"]
        )
        if not ok:
            mismatches.append(
                {
                    "ticker": ticker,
                    "date": str(date),
                    "r_next_hand": float(r_next_hand),
                    "r_next_pipeline": float(row["r_next"]),
                    "sigma_hand": float(sigma_hand),
                    "sigma_pipeline": float(row["sigma"]),
                    "label_hand": label_hand,
                    "label_pipeline": row["label"],
                }
            )

    return {"n_checked": n, "n_mismatch": len(mismatches), "mismatches": mismatches}


def truncation_test(
    raw_panel: pd.DataFrame,
    full_features: pd.DataFrame,
    k: float,
    vol_window: int,
    n: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """For `n` random (ticker, day) rows, rebuild features using only data
    truncated at that day and assert they equal the features from the full
    build. The single most important check in the repo -- also a pytest test.
    """
    from stockml.features import build_features, feature_names

    cols = feature_names()
    n = min(n, len(full_features))
    sample = full_features.sample(n=n, random_state=seed)

    mismatches = []
    for _, row in sample.iterrows():
        ticker, date = row["ticker"], pd.Timestamp(row["date"])
        truncated_raw = raw_panel[
            (raw_panel["ticker"] == ticker) & (pd.to_datetime(raw_panel["date"]) <= date)
        ]
        rebuilt = build_features(truncated_raw, k=k, vol_window=vol_window)
        rebuilt_row = rebuilt[pd.to_datetime(rebuilt["date"]) == date]
        if rebuilt_row.empty:
            mismatches.append({"ticker": ticker, "date": str(date), "reason": "rebuild produced no row"})
            continue
        rebuilt_row = rebuilt_row.iloc[0]

        row_mismatches = {}
        for c in cols:
            a, b = rebuilt_row[c], row[c]
            both_nan = pd.isna(a) and pd.isna(b)
            if not both_nan and not np.isclose(a, b, atol=1e-8, equal_nan=True):
                row_mismatches[c] = {"full_build": float(b) if pd.notna(b) else None,
                                      "truncated_build": float(a) if pd.notna(a) else None}
        if row_mismatches:
            mismatches.append({"ticker": ticker, "date": str(date), "columns": row_mismatches})

    return {"n_checked": n, "n_mismatch": len(mismatches), "mismatches": mismatches}


def run_leak_diagnostics(
    df: pd.DataFrame,
    model: Any,
    feature_cols: list[str],
    raw_panel: pd.DataFrame,
    cache_dir: str | Path,
    k: float,
    vol_window: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Run all three leak diagnostics and bundle the results."""
    return {
        "shift_test": shift_test(df, model, feature_cols),
        "label_alignment_audit": label_alignment_audit(df, cache_dir, k, vol_window, seed=seed),
        "truncation_test": truncation_test(raw_panel, df, k, vol_window, seed=seed),
    }
