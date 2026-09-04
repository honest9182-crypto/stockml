"""Ties data -> labels -> features -> split -> models -> evaluate together.

Every run writes its resolved config, metrics, and predictions to
`runs/<timestamp>_<name>/` (CLAUDE.md principle 4: reproducible).
"""

from __future__ import annotations

import copy
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import yaml

from stockml import data as data_mod
from stockml import evaluate as eval_mod
from stockml import labels as labels_mod
from stockml import split as split_mod
from stockml.features import build_features_panel, feature_names
from stockml.models.base import PerTickerModel
from stockml.models.sklearn_models import MODEL_REGISTRY
from stockml.update import Frozen, UpdatePolicy
from stockml.walk_forward import walk_forward_single_model

MANDATORY_BASELINES = ["majority_class", "always_up"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Lets `runs_dir` and `data.cache_dir` be overridden by environment
    variables, in place, without editing the config file itself.

    This exists for kaggle/stage.ipynb (see README's "Running on Kaggle"):
    a notebook's working directory is ephemeral, so it needs run output to
    land at a fixed absolute path (`/kaggle/working/runs`) regardless of
    where the repo happens to be cloned, and the price cache to be read
    straight from a mounted, read-only input dataset -- neither of which a
    config file shared with local runs should have to hardcode. Called by
    both `run.load_config` and `evolution.loop.load_evo_config` so it
    applies uniformly to step-1 and evolution configs alike.
    """
    runs_dir = os.environ.get("STOCKML_RUNS_DIR")
    if runs_dir:
        cfg["runs_dir"] = runs_dir
    cache_dir = os.environ.get("STOCKML_CACHE_DIR")
    if cache_dir:
        cfg.setdefault("data", {})["cache_dir"] = cache_dir
    return cfg


def parse_config(path: str | Path) -> dict[str, Any]:
    """Parse a config YAML and fill in defaults -- no env-var overrides.

    This is the pristine, portable version of a config: it's what a fresh
    `run()` persists into its own `runs/<ts>_<name>/config.yaml` snapshot,
    so that snapshot never bakes in a Kaggle session's env-var redirection
    (see `apply_env_overrides`) -- a run produced on Kaggle must still be
    resumable/vault-able locally, with no env vars set at all, by re-reading
    that same snapshot and getting `data/cache`/`runs` straight back.
    `load_config` (below) is what almost everything else should call.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("seed", 0)
    cfg.setdefault("per_ticker", False)
    cfg.setdefault("rolling_window", 60)
    cfg.setdefault("runs_dir", "runs")
    models = list(cfg.get("models", []))
    for b in MANDATORY_BASELINES:
        if b not in models:
            models.insert(0, b)
    cfg["models"] = models
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    """`parse_config` plus `STOCKML_RUNS_DIR`/`STOCKML_CACHE_DIR` overrides
    -- the fully resolved, ready-to-run config almost every caller wants.
    """
    return apply_env_overrides(parse_config(path))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_tickers(cfg: dict[str, Any]) -> list[str]:
    uni = cfg.get("universe", {})
    if "tickers" in uni:
        return list(uni["tickers"])
    universe_df = data_mod.load_universe(uni.get("csv", "data/tickers/sp500.csv"))
    return universe_df["ticker"].tolist()


def resolve_end_date(cfg: dict[str, Any]) -> str:
    end = cfg["data"].get("end")
    if end:
        return str(end)
    return datetime.now().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_dataset(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (raw_panel, labeled_features_panel) from the per-ticker price cache.

    `raw_panel` is calendar-aligned OHLCV with no labels/features -- leak
    diagnostics need it (the label-alignment audit reads raw closes; the
    truncation test rebuilds features from data truncated at each sampled day).
    `labeled_features_panel` is `raw_panel` plus label and feature columns,
    with rows lacking a valid label or a full feature window dropped.
    """
    tickers = resolve_tickers(cfg)
    cache_dir = cfg["data"]["cache_dir"]
    min_history_days = cfg["data"]["min_history_days"]
    start = cfg["data"]["start"]
    end = resolve_end_date(cfg)
    raw_panel, panel_report = data_mod.load_panel(
        tickers, cache_dir, min_history_days, start=start, end=end
    )
    print(
        f"[dataset] universe={panel_report.n_tickers_in} tickers -> "
        f"{panel_report.n_tickers_out} kept after min_history_days filter "
        f"({len(panel_report.dropped_missing_cache)} missing cache, "
        f"{len(panel_report.dropped_short_history)} too short)"
    )

    k = cfg["labels"]["k"]
    vol_window = cfg["labels"]["vol_window"]
    labeled = labels_mod.build_labels(raw_panel, k=k, vol_window=vol_window)
    featured = build_features_panel(labeled, k=k, vol_window=vol_window)
    feat_cols = feature_names()
    dataset = featured.dropna(subset=feat_cols).reset_index(drop=True)
    return raw_panel, dataset


# ---------------------------------------------------------------------------
# Walk-forward loop
# ---------------------------------------------------------------------------
#
# The actual fit/predict-day-by-day machinery lives in walk_forward.py, so
# evolution/fitness.py can reuse it verbatim -- a genome's evaluation is not
# a second implementation of this loop.


def run_walk_forward(
    dataset: pd.DataFrame,
    split_dates: split_mod.SplitDates,
    model_name: str,
    seed: int,
    per_ticker: bool,
    update_policy_factory: Callable[[], UpdatePolicy],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, Any, int]:
    dates = pd.to_datetime(dataset["date"])
    train_df = dataset[(dates >= split_dates.train_start) & (dates <= split_dates.train_end)]
    test_df = dataset[(dates >= split_dates.test_start) & (dates <= split_dates.test_end)]
    sanity_df = dataset[(dates >= split_dates.sanity_start) & (dates <= split_dates.sanity_end)]

    if not per_ticker:
        model = MODEL_REGISTRY[model_name](seed)
        policy = update_policy_factory()
        return walk_forward_single_model(
            train_df, test_df, sanity_df, model, policy, feature_cols
        )

    test_parts, sanity_parts = [], []
    models_by_ticker: dict[str, Any] = {}
    total_updates = 0
    for ticker, g_train in train_df.groupby("ticker", sort=False):
        g_test = test_df[test_df["ticker"] == ticker]
        g_sanity = sanity_df[sanity_df["ticker"] == ticker]
        model = MODEL_REGISTRY[model_name](seed)
        policy = update_policy_factory()
        tp, sp, fitted, nu = walk_forward_single_model(
            g_train, g_test, g_sanity, model, policy, feature_cols
        )
        test_parts.append(tp)
        sanity_parts.append(sp)
        models_by_ticker[ticker] = fitted
        total_updates += nu

    test_preds_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    sanity_preds_df = pd.concat(sanity_parts, ignore_index=True) if sanity_parts else pd.DataFrame()
    return test_preds_df, sanity_preds_df, PerTickerModel(models_by_ticker), total_updates


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------


def run(config_path: str | Path) -> Path:
    # cfg_to_persist (pristine) is what gets written to this run's own
    # config.yaml snapshot; cfg (env-override-applied) is what actually runs
    # -- see parse_config's docstring for why they're not the same object.
    cfg_to_persist = parse_config(config_path)
    cfg = copy.deepcopy(cfg_to_persist)
    apply_env_overrides(cfg)
    set_seed(cfg["seed"])

    raw_panel, dataset = build_dataset(cfg)
    feat_cols = feature_names()

    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(dataset["date"]).unique()))
    split_dates = split_mod.compute_split_dates(
        unique_dates,
        cfg["split"]["train_years"],
        cfg["split"]["sanity_days"],
        embargo_days=cfg["split"].get("embargo_days", 1),
    )
    dataset = dataset.copy()
    dataset["split"] = split_mod.assign_split(dataset["date"], split_dates)
    split_mod.assert_no_leakage(dataset, "split", "date")

    train_dist = labels_mod.class_distribution(dataset.loc[dataset["split"] == "train", "label"])
    test_dist = labels_mod.class_distribution(dataset.loc[dataset["split"] == "test", "label"])
    print(f"[run] train class distribution: {train_dist}")
    print(f"[run] test  class distribution: {test_dist}")
    class_balance_warnings = [f"train: {w}" for w in labels_mod.check_class_balance(train_dist)]
    class_balance_warnings += [f"test: {w}" for w in labels_mod.check_class_balance(test_dist)]
    for w in class_balance_warnings:
        print(f"[WARN] {w}")

    n_test_days = len(pd.to_datetime(dataset.loc[dataset["split"] == "test", "date"]).unique())

    per_ticker = cfg["per_ticker"]
    threshold = cfg.get("leak_alarm", {}).get("threshold", 0.60)
    min_days = cfg.get("leak_alarm", {}).get("min_days", 250)
    rolling_window = cfg["rolling_window"]

    results: dict[str, Any] = {}
    test_preds_by_model: dict[str, pd.DataFrame] = {}
    sanity_preds_by_model: dict[str, pd.DataFrame] = {}
    fitted_models: dict[str, Any] = {}
    leak_reports: dict[str, Any] = {}

    for model_name in cfg["models"]:
        print(f"[run] fitting {model_name} ...")
        test_preds, sanity_preds, fitted_model, n_updates = run_walk_forward(
            dataset, split_dates, model_name, cfg["seed"], per_ticker, Frozen, feat_cols
        )
        fitted_models[model_name] = fitted_model
        test_preds_by_model[model_name] = test_preds
        sanity_preds_by_model[model_name] = sanity_preds

        metrics = eval_mod.classification_metrics(test_preds["y_true"], test_preds["y_pred"])
        rolling = eval_mod.rolling_accuracy_series(
            test_preds["date"], test_preds["y_true"], test_preds["y_pred"], window=rolling_window
        )
        sanity_metrics = (
            eval_mod.classification_metrics(sanity_preds["y_true"], sanity_preds["y_pred"])
            if len(sanity_preds)
            else None
        )
        results[model_name] = {
            "metrics": metrics,
            "sanity_metrics": sanity_metrics,
            "n_updates": n_updates,
            "rolling_accuracy": rolling,
        }

        alarm = eval_mod.check_leak_alarm(metrics["accuracy"], n_test_days, threshold, min_days)
        if alarm:
            print(
                f"[LEAK ALARM] {model_name}: accuracy {metrics['accuracy']:.3f} "
                f"exceeds threshold {threshold:.2f} on {n_test_days} test days "
                f"(> {min_days}). Running leak diagnostics before reporting."
            )
            diag_df = dataset[dataset["split"] == "test"]
            leak_reports[model_name] = eval_mod.run_leak_diagnostics(
                diag_df, fitted_model, feat_cols, raw_panel,
                cfg["data"]["cache_dir"], cfg["labels"]["k"], cfg["labels"]["vol_window"],
                seed=cfg["seed"],
            )

    majority_preds = test_preds_by_model["majority_class"]
    majority_acc = float((majority_preds["y_true"] == majority_preds["y_pred"]).mean())
    majority_lookup = majority_preds.set_index(["date", "ticker"])["y_pred"]

    for model_name in cfg["models"]:
        test_preds = test_preds_by_model[model_name]
        n_hits = int((test_preds["y_true"] == test_preds["y_pred"]).sum())
        # Row-level binomial test: kept for comparison only, labelled for what
        # it is -- rows within one day aren't independent, so this overstates
        # significance. See evaluate.binomial_test_vs_baseline / CLAUDE.md.
        results[model_name]["row_level_overstated"] = eval_mod.binomial_test_vs_baseline(
            n_hits, len(test_preds), majority_acc
        )
        # Align the majority baseline's predictions to this model's rows by
        # (date, ticker) rather than assuming identical row order.
        aligned_baseline = pd.Series(
            test_preds.set_index(["date", "ticker"]).index.map(majority_lookup),
            index=test_preds.index,
        )
        results[model_name]["per_ticker"] = eval_mod.per_ticker_summary(
            test_preds["ticker"], test_preds["y_true"], test_preds["y_pred"], aligned_baseline
        )
        results[model_name]["day_level_test"] = eval_mod.day_level_paired_test(
            test_preds["date"], test_preds["y_true"], test_preds["y_pred"], aligned_baseline
        )
        results[model_name]["prediction_mix"] = eval_mod.prediction_mix(test_preds["y_pred"])
        results[model_name]["yearly_edge_pp"] = eval_mod.yearly_edge_table(
            test_preds["date"], test_preds["y_true"], test_preds["y_pred"], aligned_baseline
        )

    run_dir = _write_run(
        cfg, cfg_to_persist, results, test_preds_by_model, sanity_preds_by_model,
        fitted_models, leak_reports, class_balance_warnings,
    )
    return run_dir


def _write_run(
    cfg: dict[str, Any],
    cfg_to_persist: dict[str, Any],
    results: dict[str, Any],
    test_preds_by_model: dict[str, pd.DataFrame],
    sanity_preds_by_model: dict[str, pd.DataFrame],
    fitted_models: dict[str, Any],
    leak_reports: dict[str, Any],
    class_balance_warnings: list[str],
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = cfg.get("name", "run")
    run_dir = Path(cfg["runs_dir"]) / f"{ts}_{name}"
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_to_persist, f, sort_keys=False)

    def _json_default(o: Any) -> Any:
        if isinstance(o, pd.Series):
            return o.to_dict()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(f"not JSON serializable: {type(o)}")

    metrics_out = {
        m: {k: v for k, v in r.items() if k != "rolling_accuracy"} for m, r in results.items()
    }
    metrics_out["class_balance_warnings"] = class_balance_warnings
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, default=_json_default)

    test_preds_all = pd.concat(
        [df.assign(model=m) for m, df in test_preds_by_model.items()], ignore_index=True
    )
    sanity_preds_all = pd.concat(
        [df.assign(model=m) for m, df in sanity_preds_by_model.items()], ignore_index=True
    )
    test_preds_all.to_parquet(run_dir / "predictions_test.parquet")
    sanity_preds_all.to_parquet(run_dir / "predictions_sanity.parquet")

    for model_name, r in results.items():
        r["rolling_accuracy"].to_csv(run_dir / f"rolling_accuracy_{model_name}.csv", header=["accuracy"])
        eval_mod.save_rolling_accuracy_plot(
            r["rolling_accuracy"], run_dir / f"rolling_accuracy_{model_name}.png",
            title=f"{model_name}: rolling {cfg['rolling_window']}-day accuracy",
        )

    for model_name, model in fitted_models.items():
        try:
            joblib.dump(model, run_dir / "models" / f"{model_name}.joblib")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] could not save model {model_name}: {e}")

    if leak_reports:
        with open(run_dir / "leak_report.json", "w", encoding="utf-8") as f:
            json.dump(leak_reports, f, indent=2, default=_json_default)

    report_text = _render_report(cfg, results, class_balance_warnings, leak_reports)
    with open(run_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)

    print(f"[run] wrote {run_dir}")
    return run_dir


def _render_report(
    cfg: dict[str, Any],
    results: dict[str, Any],
    class_balance_warnings: list[str],
    leak_reports: dict[str, Any],
) -> str:
    lines = [f"=== stockml run: {cfg.get('name', 'run')} ===", ""]
    for w in class_balance_warnings:
        lines.append(f"[WARN] {w}")
    if class_balance_warnings:
        lines.append("")

    # NOTE: no p_value column here -- the row-level binomial test overstates
    # significance (rows within one day aren't independent) and is deliberately
    # kept out of report.txt. See metrics.json's "row_level_overstated_..."
    # key and the "day-level paired test" section below for the honest version.
    header = f"{'model':<16}{'accuracy':>10}{'bal.acc':>10}{'macro_f1':>10}{'n_test':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for model_name, r in results.items():
        m = r["metrics"]
        lines.append(
            f"{model_name:<16}{m['accuracy']:>10.4f}{m['balanced_accuracy']:>10.4f}"
            f"{m['macro_f1']:>10.4f}{m['n']:>10d}"
        )
    lines.append("")

    lines.append("day-level paired test vs. majority baseline (see CLAUDE.md -- rows within")
    lines.append("a day aren't independent, so this is the honest significance test, not the"
                  " row-level one):")
    for model_name, r in results.items():
        dl = r["day_level_test"]
        lines.append(
            f"  {model_name:<16} n_days={dl['n_days']:<6} mean_edge={dl['mean_edge_pp']:+7.3f}pp "
            f"t_p={dl['p_value']:>9.4g}  95% CI=[{dl['ci95_low_pp']:+.3f}, {dl['ci95_high_pp']:+.3f}]pp"
        )
    lines.append("")

    lines.append("prediction mix (share of down / stagnant / up the model actually predicted):")
    for model_name, r in results.items():
        mix = r["prediction_mix"]
        lines.append(
            f"  {model_name:<16} down={mix['down']:.3f} stagnant={mix['stagnant']:.3f} up={mix['up']:.3f}"
        )
    lines.append("")

    lines.append("per-year edge vs. majority baseline (mean daily-accuracy difference, points):")
    years = sorted({y for r in results.values() for y in r["yearly_edge_pp"].keys()})
    if years:
        lines.append(f"  {'model':<16}" + "".join(f"{y:>8}" for y in years))
        for model_name, r in results.items():
            ye = r["yearly_edge_pp"]
            lines.append(
                f"  {model_name:<16}"
                + "".join(f"{ye[y]:>8.3f}" if y in ye else f"{'--':>8}" for y in years)
            )
    lines.append("")

    lines.append("per-ticker accuracy (mean / median / p10 / p90 / n beating majority):")
    for model_name, r in results.items():
        pt = r["per_ticker"]
        lines.append(
            f"  {model_name:<16} mean={pt['mean']:.4f} median={pt['median']:.4f} "
            f"p10={pt['p10']:.4f} p90={pt['p90']:.4f} "
            f"beating_majority={pt['n_beating_majority_baseline']}/{pt['n_tickers']}"
        )
    lines.append("")

    lines.append("sanity slice (final 10 trading days -- not merged into test, not significant):")
    for model_name, r in results.items():
        sm = r["sanity_metrics"]
        if sm is None:
            lines.append(f"  {model_name:<16} (no sanity rows)")
        else:
            lines.append(f"  {model_name:<16} accuracy={sm['accuracy']:.4f} n={sm['n']}")
    lines.append("")

    if leak_reports:
        lines.append("[LEAK ALARM FIRED] see leak_report.json for full diagnostics. Summary:")
        for model_name, diag in leak_reports.items():
            st = diag["shift_test"]
            la = diag["label_alignment_audit"]
            tt = diag["truncation_test"]
            lines.append(
                f"  {model_name}: shift_test drop={st['drop']:.4f} "
                f"(orig={st['original_accuracy']:.4f} shifted={st['shifted_accuracy']:.4f}); "
                f"label_alignment mismatches={la['n_mismatch']}/{la['n_checked']}; "
                f"truncation mismatches={tt['n_mismatch']}/{tt['n_checked']}"
            )
        lines.append("")

    return "\n".join(lines)
