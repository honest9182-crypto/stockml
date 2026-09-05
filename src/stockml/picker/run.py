"""Ties the picker together: build the dataset exactly as step 1 does, fit
each model picker and compute every baseline over the same test/sanity
slices, evaluate all of them (mandatory baselines, hit mix, return check,
concentration, sector mix, the `n_picks` sweep), and write
`runs/<timestamp>_<name>/` -- CLAUDE.md's "Up-only picker" section.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from stockml import data as data_mod
from stockml import evaluate as eval_mod
from stockml import run as run_mod
from stockml import split as split_mod
from stockml.features import feature_names
from stockml.picker.baselines import DETERMINISTIC_BASELINES, random_baseline_edges
from stockml.picker.evaluate import (
    PickerResult,
    concentration_stats,
    evaluate_picker,
    k_sweep,
    per_year_edge_table,
    return_check,
    rolling_edge_series,
    save_k_sweep_plot,
    save_rolling_edge_plot,
    sector_mix,
)
from stockml.picker.scores import build_score_model, compute_scores
from stockml.picker.select import n_picks_shortfall_days, select_top_k

DEFAULT_N_BOOT = 2000
DEFAULT_LEAK_EDGE_THRESHOLD_PP = 15.0
DEFAULT_LEAK_MIN_DAYS = 250


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def parse_picker_config(path: str | Path) -> dict[str, Any]:
    """Same shape as `run.parse_config` (step 1) plus the picker's own
    fields -- pristine, no env-var overrides. See `run.parse_config`'s
    docstring for why fresh runs persist this, not `load_picker_config`'s
    override-applied result.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("seed", 0)
    cfg.setdefault("runs_dir", "runs")
    cfg.setdefault("n_picks", 10)
    cfg.setdefault("k_sweep", [1, 5, 10, 25, 50, 100])
    cfg.setdefault("score_source", "binary")
    cfg.setdefault("models", ["logreg", "hgb", "xgb"])
    cfg.setdefault("random_draws", 200)
    la = cfg.setdefault("leak_alarm", {})
    la.setdefault("edge_threshold_pp", DEFAULT_LEAK_EDGE_THRESHOLD_PP)
    la.setdefault("min_days", DEFAULT_LEAK_MIN_DAYS)
    return cfg


def load_picker_config(path: str | Path) -> dict[str, Any]:
    return run_mod.apply_env_overrides(parse_picker_config(path))


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------


def run(config_path: str | Path) -> Path:
    cfg_to_persist = parse_picker_config(config_path)
    cfg = copy.deepcopy(cfg_to_persist)
    run_mod.apply_env_overrides(cfg)
    run_mod.set_seed(cfg["seed"])
    seed = cfg["seed"]
    n_picks = cfg["n_picks"]
    score_source = cfg["score_source"]

    raw_panel, dataset = run_mod.build_dataset(cfg)
    feat_cols = feature_names()

    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(dataset["date"]).unique()))
    split_dates = split_mod.compute_split_dates(
        unique_dates, cfg["split"]["train_years"], cfg["split"]["sanity_days"],
        embargo_days=cfg["split"].get("embargo_days", 1),
    )
    dataset = dataset.copy()
    dataset["split"] = split_mod.assign_split(dataset["date"], split_dates)
    split_mod.assert_no_leakage(dataset, "split", "date")

    train_df = dataset[dataset["split"] == "train"]
    test_df = dataset[dataset["split"] == "test"]
    sanity_df = dataset[dataset["split"] == "sanity"]
    n_test_days = int(test_df["date"].nunique())
    print(f"[pick] train={len(train_df)} rows, test={len(test_df)} rows ({n_test_days} days), sanity={len(sanity_df)} rows")

    scores_by_name: dict[str, pd.DataFrame] = {}
    sanity_scores_by_name: dict[str, pd.DataFrame] = {}
    fitted_models: dict[str, Any] = {}

    for model_name in cfg["models"]:
        print(f"[pick] fitting {model_name} ({score_source}) ...")
        model = build_score_model(model_name, seed, score_source)
        test_scores, sanity_scores, fitted = compute_scores(model, train_df, test_df, sanity_df, feat_cols)
        scores_by_name[model_name] = test_scores
        sanity_scores_by_name[model_name] = sanity_scores
        fitted_models[model_name] = fitted

    for name, fn in DETERMINISTIC_BASELINES.items():
        scores_by_name[name] = fn(train_df, test_df)
        sanity_scores_by_name[name] = fn(train_df, sanity_df)

    shortfalls = n_picks_shortfall_days(test_df, n_picks)
    if shortfalls:
        print(f"[pick][WARN] {len(shortfalls)} day(s) had fewer than n_picks={n_picks} tickers: {shortfalls}")

    print(f"[pick] random baseline: {cfg['random_draws']} draws ...")
    random_edges = random_baseline_edges(test_df, n_picks, cfg["random_draws"], seed)
    random_band = (float(np.percentile(random_edges, 2.5)), float(np.percentile(random_edges, 97.5)))
    random_mean = float(np.mean(random_edges))

    results: dict[str, PickerResult] = {}
    sanity_results: dict[str, PickerResult] = {}
    picks_by_name: dict[str, pd.DataFrame] = {}
    return_checks: dict[str, dict[str, Any]] = {}
    concentration_by_name: dict[str, dict[str, Any]] = {}
    sector_mix_by_name: dict[str, pd.DataFrame] = {}
    yearly_edge_by_name: dict[str, dict[int, float]] = {}
    leak_reports: dict[str, Any] = {}

    ticker_sectors = None
    try:
        universe_table = data_mod.load_universe(cfg["universe"].get("csv", "data/tickers/sp500.csv"))
        ticker_sectors = universe_table.set_index("ticker")["sector"]
    except Exception as e:  # noqa: BLE001 -- sector mix is a nice-to-have, never fatal
        print(f"[pick][WARN] could not load ticker sectors for sector_mix: {e}")

    for name, scores_df in scores_by_name.items():
        all_days = pd.to_datetime(scores_df["date"]).unique()
        result = evaluate_picker(scores_df, all_days, n_picks, seed, n_boot=DEFAULT_N_BOOT)
        results[name] = result

        rng = np.random.default_rng(seed)
        picks = select_top_k(scores_df, n_picks, rng)
        picks_by_name[name] = picks

        sanity_scores = sanity_scores_by_name[name]
        if len(sanity_scores):
            sanity_days = pd.to_datetime(sanity_scores["date"]).unique()
            sanity_results[name] = evaluate_picker(sanity_scores, sanity_days, n_picks, seed, n_boot=0)

        return_checks[name] = return_check(picks, scores_df, n_boot=DEFAULT_N_BOOT, seed=seed)
        concentration_by_name[name] = concentration_stats(picks)
        yearly_edge_by_name[name] = per_year_edge_table(picks, scores_df)
        if ticker_sectors is not None:
            sector_mix_by_name[name] = sector_mix(picks, scores_df, ticker_sectors)

        if name in fitted_models:
            alarm = eval_mod.check_leak_alarm(
                result.mean_edge_pp, result.n_days,
                cfg["leak_alarm"]["edge_threshold_pp"], cfg["leak_alarm"]["min_days"],
            )
            if alarm:
                print(
                    f"[LEAK ALARM] {name}: mean edge {result.mean_edge_pp:.2f}pp exceeds threshold "
                    f"{cfg['leak_alarm']['edge_threshold_pp']:.1f}pp on {result.n_days} days "
                    f"(> {cfg['leak_alarm']['min_days']}). Running leak diagnostics before reporting."
                )
                diag_df = test_df
                leak_reports[name] = eval_mod.run_leak_diagnostics(
                    diag_df, fitted_models[name], feat_cols, raw_panel,
                    cfg["data"]["cache_dir"], cfg["labels"]["k"], cfg["labels"]["vol_window"], seed=seed,
                )

    # k-sweep: every model picker (the actual thing n_picks was chosen for),
    # plus every deterministic baseline, all in one table/plot -- real
    # ranking skill rises as k shrinks, noise stays flat (CLAUDE.md).
    sweep_frames = []
    for name, scores_df in scores_by_name.items():
        sweep = k_sweep(scores_df, cfg["k_sweep"], seed, n_boot=200)
        sweep["name"] = name
        sweep_frames.append(sweep)
    sweep_df = pd.concat(sweep_frames, ignore_index=True)

    run_dir = _write_run(
        cfg, cfg_to_persist, results, sanity_results, picks_by_name, return_checks,
        concentration_by_name, sector_mix_by_name, yearly_edge_by_name, scores_by_name,
        sweep_df, random_edges, random_band, random_mean, leak_reports, shortfalls,
    )
    return run_dir


def _write_run(
    cfg: dict[str, Any],
    cfg_to_persist: dict[str, Any],
    results: dict[str, PickerResult],
    sanity_results: dict[str, PickerResult],
    picks_by_name: dict[str, pd.DataFrame],
    return_checks: dict[str, dict[str, Any]],
    concentration_by_name: dict[str, dict[str, Any]],
    sector_mix_by_name: dict[str, pd.DataFrame],
    yearly_edge_by_name: dict[str, dict[int, float]],
    scores_by_name: dict[str, pd.DataFrame],
    sweep_df: pd.DataFrame,
    random_edges: np.ndarray,
    random_band: tuple[float, float],
    random_mean: float,
    leak_reports: dict[str, Any],
    shortfalls: dict[str, int],
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = cfg.get("name", "picker")
    run_dir = Path(cfg["runs_dir"]) / f"{ts}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_to_persist, f, sort_keys=False)

    picks_all = pd.concat(
        [df.assign(name=n) for n, df in picks_by_name.items()], ignore_index=True
    )
    picks_all.to_parquet(run_dir / "picks.parquet")

    sweep_df.to_csv(run_dir / "k_sweep.csv", index=False)
    save_k_sweep_plot(sweep_df, run_dir / "k_sweep.png", title=f"{name}: mean edge vs. n_picks")

    if sector_mix_by_name:
        sector_all = pd.concat(
            [df.assign(name=n) for n, df in sector_mix_by_name.items()]
        ).reset_index().rename(columns={"index": "sector"})
        sector_all.to_csv(run_dir / "sector_mix.csv", index=False)

    for picker_name, picks in picks_by_name.items():
        rolling = rolling_edge_series(picks, scores_by_name[picker_name])
        save_rolling_edge_plot(
            rolling, run_dir / f"rolling_edge_{picker_name}.png",
            title=f"{picker_name}: rolling 60-day edge vs. base rate",
        )

    def _json_default(o: Any) -> Any:
        if isinstance(o, pd.Series):
            return o.to_dict()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serializable: {type(o)}")

    metrics_out = {
        "n_picks": cfg["n_picks"],
        "score_source": cfg["score_source"],
        "results": {n: r.to_dict() for n, r in results.items()},
        "sanity_results": {n: r.to_dict() for n, r in sanity_results.items()},
        "return_checks": return_checks,
        "concentration": concentration_by_name,
        "yearly_edge_pp": yearly_edge_by_name,
        "random_baseline": {
            "n_draws": len(random_edges), "mean_edge_pp": random_mean,
            "ci95_low_pp": random_band[0], "ci95_high_pp": random_band[1],
        },
        "shortfall_days": shortfalls,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, default=_json_default)

    if leak_reports:
        with open(run_dir / "leak_report.json", "w", encoding="utf-8") as f:
            json.dump(leak_reports, f, indent=2, default=_json_default)

    report_text = _render_report(
        cfg, results, sanity_results, return_checks, concentration_by_name,
        yearly_edge_by_name, sweep_df, random_band, random_mean, leak_reports, shortfalls,
    )
    with open(run_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)

    print(f"[pick] wrote {run_dir}")
    return run_dir


def _render_report(
    cfg: dict[str, Any],
    results: dict[str, PickerResult],
    sanity_results: dict[str, PickerResult],
    return_checks: dict[str, dict[str, Any]],
    concentration_by_name: dict[str, dict[str, Any]],
    yearly_edge_by_name: dict[str, dict[int, float]],
    sweep_df: pd.DataFrame,
    random_band: tuple[float, float],
    random_mean: float,
    leak_reports: dict[str, Any],
    shortfalls: dict[str, int],
) -> str:
    lines = [f"=== stockml pick: {cfg.get('name', 'picker')} (n_picks={cfg['n_picks']}, score_source={cfg['score_source']}) ==="]
    lines.append("")
    if shortfalls:
        lines.append(f"[WARN] {len(shortfalls)} day(s) had fewer than n_picks tickers available: {shortfalls}")
        lines.append("")

    lines.append(
        f"random baseline (luck band, {cfg['random_draws']} draws): "
        f"mean={random_mean:+.3f}pp  95% band=[{random_band[0]:+.3f}, {random_band[1]:+.3f}]pp"
    )
    lines.append("")

    model_names = list(cfg["models"])
    baseline_names = [n for n in results if n not in model_names]
    lines.append("picker/baseline results (mean precision vs. mean base rate; edge = the difference):")
    for n in model_names + baseline_names:
        if n not in results:
            continue
        r = results[n]
        beats_top_vol = ""
        if n != "top_vol" and "top_vol" in results:
            beats_top_vol = "  [beats top_vol]" if r.ci95_low_pp > results["top_vol"].mean_edge_pp else ""
        beats_random = "  [beats random band]" if r.ci95_low_pp > random_band[1] else ""
        lines.append(
            f"  {n:<12} precision={r.mean_precision:.4f} base_rate={r.mean_base_rate:.4f} "
            f"mean_edge={r.mean_edge_pp:+.3f}pp se={r.se_pp:.3f}pp "
            f"95%CI=[{r.ci95_low_pp:+.3f},{r.ci95_high_pp:+.3f}]pp n_days={r.n_days}"
            f"{beats_top_vol}{beats_random}"
        )
    lines.append("")

    lines.append("hit mix (share of picks whose true label was down/stagnant/up):")
    for n in model_names + baseline_names:
        if n not in results:
            continue
        mix = results[n].hit_mix
        lines.append(f"  {n:<12} down={mix['down']:.3f} stagnant={mix['stagnant']:.3f} up={mix['up']:.3f}")
    lines.append("")

    lines.append('return check (secondary -- "not a P&L"; mean r_next of picks minus universe, bp):')
    for n in model_names + baseline_names:
        if n not in return_checks:
            continue
        rc = return_checks[n]
        lines.append(
            f"  {n:<12} mean_edge={rc['mean_edge_bp']:+.2f}bp t_p={rc['p_value']:.4g} "
            f"95%CI=[{rc['ci95_low_bp']:+.2f},{rc['ci95_high_bp']:+.2f}]bp n_days={rc['n_days']}"
        )
    lines.append("")

    lines.append("concentration (distinct tickers ever picked / top-10 share of pick-slots / mean day-to-day overlap):")
    for n in model_names + baseline_names:
        if n not in concentration_by_name:
            continue
        c = concentration_by_name[n]
        lines.append(
            f"  {n:<12} n_distinct={c['n_distinct_tickers']} top10_share={c['top_n_share']:.3f} "
            f"overlap={c['mean_consecutive_day_overlap']:.3f}"
        )
    lines.append("")

    lines.append("per-year mean edge vs. base rate (points):")
    years = sorted({y for ye in yearly_edge_by_name.values() for y in ye})
    if years:
        lines.append(f"  {'name':<12}" + "".join(f"{y:>8}" for y in years))
        for n in model_names + baseline_names:
            if n not in yearly_edge_by_name:
                continue
            ye = yearly_edge_by_name[n]
            lines.append(f"  {n:<12}" + "".join(f"{ye[y]:>8.3f}" if y in ye else f"{'--':>8}" for y in years))
    lines.append("")

    lines.append(f"n_picks sweep ({cfg['k_sweep']}): mean edge (pp) by k -- see k_sweep.csv/k_sweep.png:")
    for n in model_names + baseline_names:
        sub = sweep_df[sweep_df["name"] == n].sort_values("k")
        if sub.empty:
            continue
        vals = "  ".join(f"k={int(row.k)}:{row.mean_edge_pp:+.3f}pp" for row in sub.itertuples())
        lines.append(f"  {n:<12} {vals}")
    lines.append("")

    lines.append("sanity slice (final trading days -- not merged into test, not significant):")
    for n in model_names + baseline_names:
        if n not in sanity_results:
            lines.append(f"  {n:<12} (no sanity rows)")
            continue
        sr = sanity_results[n]
        lines.append(
            f"  {n:<12} precision={sr.mean_precision:.4f} base_rate={sr.mean_base_rate:.4f} "
            f"mean_edge={sr.mean_edge_pp:+.3f}pp n_days={sr.n_days}"
        )
    lines.append("")

    if leak_reports:
        threshold = cfg["leak_alarm"]["edge_threshold_pp"]
        lines.append(f"[LEAK ALARM FIRED] (edge > {threshold:.1f}pp) see leak_report.json. Summary:")
        for n, diag in leak_reports.items():
            st = diag["shift_test"]
            la = diag["label_alignment_audit"]
            tt = diag["truncation_test"]
            lines.append(
                f"  {n}: shift_test drop={st['drop']:.4f} "
                f"(orig={st['original_accuracy']:.4f} shifted={st['shifted_accuracy']:.4f}); "
                f"label_alignment mismatches={la['n_mismatch']}/{la['n_checked']}; "
                f"truncation mismatches={tt['n_mismatch']}/{tt['n_checked']}"
            )
        lines.append("")

    return "\n".join(lines)
