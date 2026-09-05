"""Scores a picker's picks against the day's own base rate, and the extra
diagnostics CLAUDE.md's "Up-only picker" section requires: hit mix,
concentration, a return check, and the `n_picks` sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stockml import evaluate as eval_mod
from stockml.picker.select import select_top_k

DEFAULT_PICKER_N_BOOT = 200


def daily_precision_and_base(picks_df: pd.DataFrame, universe_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """`precision_d` = share of that day's picks whose true label is `up`.
    `base_d` = share of *every* ticker with a row that day (`universe_df`,
    not the picks) whose true label is `up` -- exactly the expected
    precision of `n_picks` random picks that day. Both indexed by date.
    """
    precision_d = picks_df.groupby("date")["y_true"].apply(lambda s: (s == "up").mean()).sort_index()
    base_d = universe_df.groupby("date")["y_true"].apply(lambda s: (s == "up").mean()).sort_index()
    return precision_d, base_d


@dataclass(frozen=True)
class PickerResult:
    """Same shape as `evolution.fitness.FitnessResult`, on purpose -- see
    CLAUDE.md's "Evolution-ready, not evolution-wired" note.
    """

    n_days: int
    mean_precision: float
    mean_base_rate: float
    mean_edge_pp: float
    se_pp: float
    fitness: float  # mean_edge_pp - se_pp -- the selection scalar, matching FitnessResult
    ci95_low_pp: float
    ci95_high_pp: float
    hit_mix: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_days": self.n_days,
            "mean_precision": self.mean_precision,
            "mean_base_rate": self.mean_base_rate,
            "mean_edge_pp": self.mean_edge_pp,
            "se_pp": self.se_pp,
            "fitness": self.fitness,
            "ci95_low_pp": self.ci95_low_pp,
            "ci95_high_pp": self.ci95_high_pp,
            "hit_mix": self.hit_mix,
        }


def evaluate_picker(
    scores_df: pd.DataFrame,
    zone_days: Any,
    n_picks: int,
    seed: int,
    n_boot: int = DEFAULT_PICKER_N_BOOT,
) -> PickerResult:
    """Selects `n_picks` per day (seeded) over `zone_days` and scores the
    result: mean precision, mean base rate, mean edge, a one-sided t-test
    and 20-day block-bootstrap CI on the daily edge (via
    `evaluate.summarize_daily_series` -- the same treatment
    `day_level_paired_test` gets, see CLAUDE.md), and the picks' hit mix.

    `zone_days` restricts `scores_df` to a subset of dates first -- so this
    same function works whether `scores_df` spans one zone already (the
    normal `pick` run) or several (evolution's arena/vault zones later):
    the caller decides what "a zone" means by which dates it passes.
    """
    zone_dates = pd.to_datetime(pd.Index(list(zone_days)))
    dates = pd.to_datetime(scores_df["date"])
    zone_scores = scores_df[dates.isin(zone_dates)]

    rng = np.random.default_rng(seed)
    picks = select_top_k(zone_scores, n_picks, rng)

    precision_d, base_d = daily_precision_and_base(picks, zone_scores)
    edge = (precision_d - base_d).dropna()
    summary = eval_mod.summarize_daily_series(edge, n_boot=n_boot, seed=seed)

    hit_mix = eval_mod.prediction_mix(picks["y_true"])
    fitness = summary["mean_edge_pp"] - summary["se_pp"]

    return PickerResult(
        n_days=summary["n_days"],
        mean_precision=float(precision_d.reindex(edge.index).mean()) if len(edge) else 0.0,
        mean_base_rate=float(base_d.reindex(edge.index).mean()) if len(edge) else 0.0,
        mean_edge_pp=summary["mean_edge_pp"],
        se_pp=summary["se_pp"],
        fitness=fitness,
        ci95_low_pp=summary["ci95_low_pp"],
        ci95_high_pp=summary["ci95_high_pp"],
        hit_mix=hit_mix,
    )


# ---------------------------------------------------------------------------
# Per-year / rolling edge (step-1's yearly_edge_table/rolling_accuracy_series,
# for the picker's precision-vs-base-rate edge instead of accuracy)
# ---------------------------------------------------------------------------


def per_year_edge_table(picks_df: pd.DataFrame, universe_df: pd.DataFrame) -> dict[int, float]:
    precision_d, base_d = daily_precision_and_base(picks_df, universe_df)
    edge = (precision_d - base_d).dropna()
    by_year = edge.groupby(edge.index.year).mean() * 100
    return {int(y): float(v) for y, v in by_year.items()}


def rolling_edge_series(picks_df: pd.DataFrame, universe_df: pd.DataFrame, window: int = 60) -> pd.Series:
    precision_d, base_d = daily_precision_and_base(picks_df, universe_df)
    edge = (precision_d - base_d).dropna()
    return edge.rolling(window=window, min_periods=max(1, window // 3)).mean()


def save_rolling_edge_plot(series: pd.Series, path: str | Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.to_numpy() * 100)
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1, label="0 (no edge)")
    ax.set_title(title)
    ax.set_ylabel("rolling edge (pp)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Return check (secondary -- "not a P&L", CLAUDE.md)
# ---------------------------------------------------------------------------


def return_check(
    picks_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    block_size: int = 20,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Mean `r_next` of the picks minus the mean `r_next` of every ticker
    that day, in basis points, with the same daily-series test as
    everything else. Equal weight, no costs, nothing path-dependent --
    volatility alone cannot move this number (unlike precision, which a
    volatile-but-directionless pick can inflate), so it either corroborates
    the precision edge or contradicts it.
    """
    picks_r_next_d = picks_df.groupby("date")["r_next"].mean()
    universe_r_next_d = universe_df.groupby("date")["r_next"].mean()
    edge_fraction = (picks_r_next_d - universe_r_next_d).dropna()
    summary = eval_mod.summarize_daily_series(edge_fraction, block_size=block_size, n_boot=n_boot, seed=seed)
    # summarize_daily_series' *_pp fields are edge_fraction * 100 (percentage
    # points); 1 percentage point = 100 basis points, so *100 again for bp.
    return {
        "n_days": summary["n_days"],
        "mean_edge_bp": summary["mean_edge_pp"] * 100,
        "se_bp": summary["se_pp"] * 100,
        "t_stat": summary["t_stat"],
        "p_value": summary["p_value"],
        "ci95_low_bp": summary["ci95_low_pp"] * 100,
        "ci95_high_bp": summary["ci95_high_pp"] * 100,
    }


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------


def concentration_stats(picks_df: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
    """Distinct tickers ever picked, the share of all pick-slots taken by
    the `top_n` most-picked tickers (named), and the mean overlap between
    consecutive days' pick sets (turnover, 1.0 = identical set both days).
    A picker that names the same tickers every day has found a pattern in
    the *names*, not the days -- this is what would show it.
    """
    counts = picks_df["ticker"].value_counts()
    top = counts.head(top_n)
    total_slots = int(len(picks_df))
    top_share = float(top.sum() / total_slots) if total_slots else 0.0

    by_day = picks_df.groupby("date")["ticker"].apply(set).sort_index()
    overlaps = []
    prev: set | None = None
    for _, tickers in by_day.items():
        if prev is not None and len(tickers) and len(prev):
            overlaps.append(len(tickers & prev) / max(len(tickers), len(prev)))
        prev = tickers

    return {
        "n_distinct_tickers": int(len(counts)),
        "top_n": top_n,
        "top_n_share": top_share,
        "top_n_tickers": {str(t): int(c) for t, c in top.items()},
        "mean_consecutive_day_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
    }


def sector_mix(picks_df: pd.DataFrame, universe_df: pd.DataFrame, ticker_sectors: pd.Series) -> pd.DataFrame:
    """`ticker_sectors`: a ticker -> sector `Series` (e.g.
    `data.load_universe(...).set_index("ticker")["sector"]`). Returns a
    DataFrame indexed by sector with `pick_share`/`universe_share` columns,
    sorted by pick share descending.
    """
    # .map() preserves the calling series' own name ("ticker"), not the
    # mapped-to semantics -- rename explicitly so the resulting index (and
    # the CSV column it becomes after reset_index()) says "sector", not
    # "ticker".
    pick_sectors = picks_df["ticker"].map(ticker_sectors).fillna("unknown").rename("sector")
    universe_sectors = universe_df["ticker"].map(ticker_sectors).fillna("unknown").rename("sector")
    pick_share = pick_sectors.value_counts(normalize=True)
    universe_share = universe_sectors.value_counts(normalize=True)
    out = pd.DataFrame({"pick_share": pick_share, "universe_share": universe_share}).fillna(0.0)
    out.index.name = "sector"
    return out.sort_values("pick_share", ascending=False)


# ---------------------------------------------------------------------------
# n_picks sweep
# ---------------------------------------------------------------------------


def k_sweep(
    scores_df: pd.DataFrame, k_values: list[int], seed: int, n_boot: int = DEFAULT_PICKER_N_BOOT
) -> pd.DataFrame:
    """Re-runs selection (the scores don't change) for each `k` in
    `k_values` over the whole of `scores_df`, via `evaluate_picker`.
    Real ranking skill shows up as an edge that *rises* as `k` shrinks;
    noise shows up flat (CLAUDE.md).
    """
    all_days = pd.to_datetime(scores_df["date"]).unique()
    rows = []
    for k in k_values:
        result = evaluate_picker(scores_df, all_days, k, seed, n_boot=n_boot)
        rows.append(
            {
                "k": k,
                "mean_edge_pp": result.mean_edge_pp,
                "se_pp": result.se_pp,
                "ci95_low_pp": result.ci95_low_pp,
                "ci95_high_pp": result.ci95_high_pp,
                "n_days": result.n_days,
            }
        )
    return pd.DataFrame(rows)


def save_k_sweep_plot(sweep_df: pd.DataFrame, path: str | Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        sweep_df["k"], sweep_df["mean_edge_pp"],
        yerr=[
            sweep_df["mean_edge_pp"] - sweep_df["ci95_low_pp"],
            sweep_df["ci95_high_pp"] - sweep_df["mean_edge_pp"],
        ],
        fmt="o-", capsize=3,
    )
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1, label="0 (no edge)")
    ax.set_xlabel("n_picks (k)")
    ax.set_ylabel("mean edge (pp), 95% CI")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
