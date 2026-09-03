"""Command-line entry points: download, dataset, run, report, leakcheck."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import typer
import yaml

from stockml import data as data_mod
from stockml import evaluate as eval_mod
from stockml import labels as labels_mod
from stockml import run as run_mod
from stockml import split as split_mod
from stockml.features import feature_names

app = typer.Typer(add_completion=False, help="stockml: honest next-day direction research framework.")


@app.command()
def download(
    config: str = typer.Option(..., "--config", help="Path to a run config YAML."),
    refresh: bool = typer.Option(False, "--refresh", help="Re-download even if cached."),
) -> None:
    """Download and cache daily OHLCV for the config's ticker universe."""
    cfg = run_mod.load_config(config)
    tickers = run_mod.resolve_tickers(cfg)
    end = run_mod.resolve_end_date(cfg)
    typer.echo(f"[download] {len(tickers)} tickers, {cfg['data']['start']} -> {end}, refresh={refresh}")
    report = data_mod.download_prices(
        tickers, cfg["data"]["start"], end, cfg["data"]["cache_dir"], refresh=refresh
    )
    typer.echo(f"[download] {report.summary()}")
    if report.failed:
        typer.echo(f"[download] failed tickers: {report.failed}")


@app.command()
def dataset(
    config: str = typer.Option(..., "--config", help="Path to a run config YAML."),
) -> None:
    """Build features + labels from the cached prices and print class balance."""
    cfg = run_mod.load_config(config)
    _raw_panel, ds = run_mod.build_dataset(cfg)
    typer.echo(f"[dataset] {len(ds)} rows, {ds['ticker'].nunique()} tickers")
    dist = labels_mod.class_distribution(ds["label"])
    typer.echo(f"[dataset] class distribution: {dist}")
    for w in labels_mod.check_class_balance(dist):
        typer.echo(f"[WARN] {w}")


@app.command()
def run(
    config: str = typer.Option(..., "--config", help="Path to a run config YAML."),
) -> None:
    """Run the walk-forward pipeline end to end and write runs/<ts>_<name>/."""
    run_dir = run_mod.run(config)
    typer.echo(f"[run] done -> {run_dir}")


@app.command()
def report(
    run_dir: str = typer.Argument(..., help="Path to a runs/<ts>_<name>/ directory."),
) -> None:
    """Pretty-print a completed run's results table + baselines + warnings."""
    p = Path(run_dir)
    report_path = p / "report.txt"
    if not report_path.exists():
        typer.echo(f"no report.txt found in {p}", err=True)
        raise typer.Exit(code=1)
    typer.echo(report_path.read_text(encoding="utf-8"))
    leak_path = p / "leak_report.json"
    if leak_path.exists():
        typer.echo(f"[report] leak diagnostics were run for this run -- see {leak_path}")


@app.command()
def leakcheck(
    run_dir: str = typer.Argument(..., help="Path to a runs/<ts>_<name>/ directory."),
) -> None:
    """Run the leak diagnostics (shift test, label audit, truncation test) on demand."""
    p = Path(run_dir)
    with open(p / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_panel, dataset_df = run_mod.build_dataset(cfg)
    feat_cols = feature_names()

    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(dataset_df["date"]).unique()))
    split_dates = split_mod.compute_split_dates(
        unique_dates, cfg["split"]["train_years"], cfg["split"]["sanity_days"]
    )
    dataset_df["split"] = split_mod.assign_split(dataset_df["date"], split_dates)
    diag_df = dataset_df[dataset_df["split"] == "test"]

    models_dir = p / "models"
    leak_reports = {}
    for model_path in sorted(models_dir.glob("*.joblib")):
        model_name = model_path.stem
        typer.echo(f"[leakcheck] {model_name} ...")
        model = joblib.load(model_path)
        leak_reports[model_name] = eval_mod.run_leak_diagnostics(
            diag_df, model, feat_cols, raw_panel,
            cfg["data"]["cache_dir"], cfg["labels"]["k"], cfg["labels"]["vol_window"],
            seed=cfg.get("seed", 0),
        )
        st = leak_reports[model_name]["shift_test"]
        la = leak_reports[model_name]["label_alignment_audit"]
        tt = leak_reports[model_name]["truncation_test"]
        typer.echo(
            f"  shift_test: orig={st['original_accuracy']:.4f} shifted={st['shifted_accuracy']:.4f} "
            f"drop={st['drop']:.4f}"
        )
        typer.echo(f"  label_alignment_audit: {la['n_mismatch']}/{la['n_checked']} mismatches")
        typer.echo(f"  truncation_test: {tt['n_mismatch']}/{tt['n_checked']} mismatches")

    out_path = p / "leak_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(leak_reports, f, indent=2)
    typer.echo(f"[leakcheck] wrote {out_path}")


if __name__ == "__main__":
    app()
