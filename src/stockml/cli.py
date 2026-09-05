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
    run_mod.apply_env_overrides(cfg)

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


# ---------------------------------------------------------------------------
# Evolutionary search (step 1.5)
# ---------------------------------------------------------------------------


def _find_latest_completed_evo_run(runs_dir: str, name: str) -> Path:
    from stockml.evolution.loop import read_progress

    for p in sorted(Path(runs_dir).glob("evo_*"), reverse=True):  # dir names sort chronologically
        cfg_path = p / "config.yaml"
        if not cfg_path.exists():
            continue
        with open(cfg_path, "r", encoding="utf-8") as f:
            run_cfg = yaml.safe_load(f)
        if run_cfg.get("name") != name:
            continue
        try:
            progress = read_progress(p)
        except FileNotFoundError:
            continue
        if progress.get("status") == "completed":
            return p
    raise FileNotFoundError(
        f"no completed evolution run found under {runs_dir!r} matching config name {name!r} -- "
        f"pass --run-dir explicitly"
    )


@app.command()
def evolve(
    config: str = typer.Option(..., "--config", help="Path to an evolution config YAML."),
    quick: bool = typer.Option(False, "--quick", help="20 tickers, 6 genomes, 3 generations."),
    resume: str = typer.Option(None, "--resume", help="Path to a runs/evo_<ts>_<name>/ to resume."),
) -> None:
    """Run the evolutionary search: builds generation 0, breeds/evaluates
    each subsequent generation, writes runs/evo_<ts>_<name>/.
    """
    from stockml.evolution.loop import evolve as evolve_fn

    run_dir = evolve_fn(config, quick=quick, resume=resume)
    typer.echo(f"[evolve] done -> {run_dir}")


@app.command(name="evolve-control")
def evolve_control(
    config: str = typer.Option(..., "--config", help="Path to an evolution config YAML."),
    kind: str = typer.Option(..., "--kind", help="'random' or 'null'."),
    run_dir: str = typer.Option(
        None, "--run-dir", help="Target runs/evo_<ts>_<name>/. Default: latest completed run matching this config's name."
    ),
) -> None:
    """Run one of the two controls against a completed evolution run, at
    the same budget. Separate from `evolve` so an overnight can be split:
    evolution one night, controls the next.
    """
    from stockml.evolution.controls import run_null_control, run_random_search_control
    from stockml.evolution.fitness import canonical_majority_daily
    from stockml.evolution.loop import build_evo_dataset, load_evo_config
    from stockml.evolution.zones import compute_evo_zones

    if kind not in ("random", "null"):
        typer.echo(f"--kind must be 'random' or 'null', got {kind!r}", err=True)
        raise typer.Exit(code=1)

    # --config only locates the target run (by name) when --run-dir is
    # omitted. Everything else must use the *target run's own* saved
    # config.yaml, not a fresh load of --config -- the control has to run
    # against the exact same universe/dates/settings the run it's being
    # compared against actually used (e.g. a --quick run's 20-ticker
    # override), or its budget and fitness numbers aren't comparable at all.
    lookup_cfg = load_evo_config(config)
    target_dir = Path(run_dir) if run_dir else _find_latest_completed_evo_run(
        lookup_cfg["runs_dir"], lookup_cfg["name"]
    )
    typer.echo(f"[evolve-control {kind}] target run: {target_dir}")

    with open(target_dir / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_mod.apply_env_overrides(cfg)

    if kind == "null":
        run_null_control(cfg, target_dir)
        return

    gens = pd.read_csv(target_dir / "generations.csv")
    n_evals = int(gens["cumulative_evaluations"].iloc[-1])
    raw_panel, dataset = build_evo_dataset(cfg)
    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(dataset["date"]).unique()))
    zones = compute_evo_zones(
        unique_dates, cfg["split"]["train_years"], cfg["split"]["sanity_days"],
        cfg["split"].get("embargo_days", 1), cfg["zones"]["arena_end"], cfg["zones"]["vault_start"],
    )
    majority_arena = canonical_majority_daily(dataset, zones, "arena")
    run_random_search_control(cfg, dataset, zones, majority_arena, n_evals, target_dir)


@app.command()
def vault(
    run_dir: str = typer.Argument(..., help="Path to a runs/evo_<ts>_<name>/ directory."),
) -> None:
    """Open the vault: a one-time look at the pre-declared list of
    individuals on the untouched vault zone. Refuses if the run is still in
    progress. Every look is logged permanently -- see CLAUDE.md.
    """
    from stockml.evolution.vault import run_vault as run_vault_fn

    run_vault_fn(run_dir)


@app.command()
def lineage(
    run_dir: str = typer.Argument(..., help="Path to a runs/evo_<ts>_<name>/ directory."),
    id: str = typer.Option(..., "--id", help="Individual id, e.g. '012_003'."),
) -> None:
    """Trace one individual's ancestry back to generation 0."""
    from stockml.evolution.lineage import render_family_tree

    typer.echo(render_family_tree(run_dir, id))


@app.command(name="evo-report")
def evo_report(
    run_dir: str = typer.Argument(..., help="Path to a runs/evo_<ts>_<name>/ directory."),
) -> None:
    """Re-render generations.csv/gene_frequency.csv into the fitness,
    diversity, and feature-frequency plots plus the champion's family tree.
    Works with whichever of evolve/evolve-control/vault have finished so
    far, and says clearly which haven't.
    """
    from stockml.evolution.outputs import render_all

    render_all(run_dir)


# ---------------------------------------------------------------------------
# Up-only picker (step 1.6)
# ---------------------------------------------------------------------------


@app.command()
def pick(
    config: str = typer.Option(..., "--config", help="Path to a picker config YAML."),
) -> None:
    """Run the up-only picker walk-forward and write runs/<ts>_<name>/."""
    from stockml.picker.run import run as pick_run

    run_dir = pick_run(config)
    typer.echo(f"[pick] done -> {run_dir}")


@app.command(name="pick-report")
def pick_report(
    run_dir: str = typer.Argument(..., help="Path to a runs/<ts>_<name>/ directory from `stockml pick`."),
) -> None:
    """Re-print a completed picker run's report.txt (table, baselines, sweep, warnings)."""
    p = Path(run_dir)
    report_path = p / "report.txt"
    if not report_path.exists():
        typer.echo(f"no report.txt found in {p}", err=True)
        raise typer.Exit(code=1)
    typer.echo(report_path.read_text(encoding="utf-8"))
    leak_path = p / "leak_report.json"
    if leak_path.exists():
        typer.echo(f"[pick-report] leak diagnostics were run for this run -- see {leak_path}")


# ---------------------------------------------------------------------------
# Kaggle (see README's "Running on Kaggle")
# ---------------------------------------------------------------------------


@app.command(name="fetch-run")
def fetch_run(
    kernel_ref: str = typer.Argument(..., help="Kaggle kernel ref, e.g. 'yourusername/stockml-stage-evolve'."),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Local runs/ directory to land the run folder(s) in."),
) -> None:
    """Download a Kaggle notebook's output (`kaggle kernels output`) and
    fold its run folder(s) into runs_dir, so a run produced on Kaggle by
    kaggle/stage.ipynb (which sets STOCKML_RUNS_DIR=/kaggle/working/runs)
    can be inspected, resumed, or vaulted exactly like a local one.

    The raw download always lands whole under
    runs_dir/kernel_output/<kernel-ref>/ first; any runs/<...>/ folder
    inside it (evo_<ts>_<name>/ or <ts>_<name>/) is then copied up into
    runs_dir itself. A name collision with an existing local run is left
    where it is under kernel_output/, with a warning, rather than
    overwritten.

    Requires the `kaggle` CLI installed and configured (same as
    scripts/upload_cache_dataset.py) -- this only shells out to it.
    """
    import shutil
    import subprocess
    import sys

    runs_path = Path(runs_dir)
    slug = kernel_ref.replace("/", "_")
    raw_dest = runs_path / "kernel_output" / slug
    raw_dest.mkdir(parents=True, exist_ok=True)

    # shutil.which first (in case the console script is on PATH), else fall
    # back to `python -m kaggle` under this interpreter -- same reasoning as
    # scripts/upload_cache_dataset.py's _kaggle_cmd_prefix.
    kaggle_bin = shutil.which("kaggle")
    kaggle_prefix = [kaggle_bin] if kaggle_bin else [sys.executable, "-m", "kaggle"]
    cmd = kaggle_prefix + ["kernels", "output", kernel_ref, "-p", str(raw_dest)]
    typer.echo(f"[fetch-run] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # Observed directly (see scripts/upload_cache_dataset.py's matching
        # note): for a large output, the kaggle CLI can print every file as
        # downloaded and still exit non-zero on some final housekeeping
        # call. Don't bail here -- check what actually landed below instead.
        typer.echo(
            "[fetch-run][WARN] kaggle exited non-zero -- this can happen even when the download "
            "actually completed. Checking what was downloaded before giving up."
        )

    nested_runs = raw_dest / "runs"
    landed = []
    if nested_runs.is_dir():
        for child in sorted(nested_runs.iterdir()):
            if not child.is_dir():
                continue
            dest = runs_path / child.name
            if dest.exists():
                typer.echo(f"[fetch-run][WARN] {dest} already exists locally -- left the downloaded copy at {child}")
                continue
            shutil.copytree(child, dest)
            landed.append(dest)

    typer.echo(f"[fetch-run] raw output -> {raw_dest}")
    if landed:
        for d in landed:
            typer.echo(f"[fetch-run] run folder -> {d}")
    else:
        typer.echo(
            "[fetch-run][WARN] no runs/<...>/ folders found in the kernel output -- "
            "was STOCKML_RUNS_DIR set to /kaggle/working/runs in the notebook?"
        )


if __name__ == "__main__":
    app()
