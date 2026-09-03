"""Derived, regenerable run artifacts: `gene_frequency.csv` rows (written
incrementally by `loop.py`, one per generation) and the plots + family tree
`evo-report` (re)renders on demand. `render_all` reads only what's on disk
and is safe to call at any point in a run's life -- it marks missing
controls/vault clearly rather than failing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from stockml.evolution.genome import FEATURE_NAMES, GENE_GRIDS
from stockml.evolution.population import Individual


def gene_frequency_row(population: list[Individual], generation: int) -> dict[str, Any]:
    """One row: generation + share of the population with each feature bit
    on + share of the population at each grid gene's each possible value.
    """
    n = len(population)
    row: dict[str, Any] = {"generation": generation, "population_size": n}
    for i, name in enumerate(FEATURE_NAMES):
        on = sum(1 for ind in population if ind.genome.feature_mask[i])
        row[f"feat::{name}"] = on / n if n else 0.0
    for gene_name, grid in GENE_GRIDS.items():
        for val in grid:
            count = sum(1 for ind in population if getattr(ind.genome, gene_name) == val)
            row[f"{gene_name}::{val}"] = count / n if n else 0.0
    return row


def gene_frequency_fieldnames() -> list[str]:
    fields = ["generation", "population_size"]
    fields += [f"feat::{name}" for name in FEATURE_NAMES]
    for gene_name, grid in GENE_GRIDS.items():
        fields += [f"{gene_name}::{val}" for val in grid]
    return fields


def write_gene_frequency_row(run_dir: str | Path, row: dict[str, Any]) -> None:
    path = Path(run_dir) / "gene_frequency.csv"
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=gene_frequency_fieldnames())
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _matplotlib_axes(figsize: tuple[float, float]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    return plt, fig, ax


def _render_fitness_plot(run_dir: Path, gens: pd.DataFrame) -> None:
    from stockml.evolution import lineage as lineage_mod

    plt, fig, ax = _matplotlib_axes((10, 5))
    ax.plot(gens["cumulative_evaluations"], gens["best_fitness"], label="evolution: best", color="tab:blue")
    ax.plot(
        gens["cumulative_evaluations"], gens["mean_fitness"], label="evolution: mean",
        color="tab:blue", linestyle="--", alpha=0.6,
    )

    random_summary_path = run_dir / "control_random" / "summary.json"
    if random_summary_path.exists():
        records = lineage_mod.read_lineage(run_dir / "control_random")
        ordered = sorted(records.values(), key=lambda r: r["id"])
        best = -float("inf")
        xs, ys = [], []
        for i, rec in enumerate(ordered, start=1):
            best = max(best, rec["fitness"]["fitness"])
            xs.append(i)
            ys.append(best)
        ax.plot(xs, ys, label="random search: running best", color="tab:orange")
    else:
        ax.text(
            0.02, 0.02, "[random search control: not yet run]",
            transform=ax.transAxes, color="tab:orange", fontsize=9,
        )

    null_csv = run_dir / "control_null" / "generations.csv"
    if null_csv.exists():
        null_gens = pd.read_csv(null_csv)
        ax.plot(
            null_gens["cumulative_evaluations"], null_gens["best_fitness"],
            label="null (shuffled labels): best", color="tab:red",
        )
    else:
        ax.text(
            0.02, 0.07, "[null control: not yet run]",
            transform=ax.transAxes, color="tab:red", fontsize=9,
        )

    ax.axhline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("cumulative fitness evaluations (the actual shared budget)")
    ax.set_ylabel("fitness: mean daily edge vs. majority - 1 SE (points)")
    ax.set_title("fitness vs. budget: evolution vs. random search vs. null")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / "fitness_by_generation.png", dpi=120)
    plt.close(fig)


def _render_diversity_plot(run_dir: Path, gens: pd.DataFrame) -> None:
    plt, fig, ax = _matplotlib_axes((10, 4))
    ax.plot(gens["generation"], gens["diversity"], color="tab:green")
    storm_mask = gens["storm"].astype(str).isin(["True", "true", "1"])
    storms = gens[storm_mask]
    if len(storms):
        ax.scatter(storms["generation"], storms["diversity"], color="tab:red", zorder=5, label="storm")
    ax.set_xlabel("generation")
    ax.set_ylabel("mean pairwise genetic distance")
    ax.set_title("population diversity by generation")
    if len(storms):
        ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "diversity_by_generation.png", dpi=120)
    plt.close(fig)


def _render_feature_heatmap(run_dir: Path, gf: pd.DataFrame) -> None:
    feat_cols = [c for c in gf.columns if c.startswith("feat::")]
    if not feat_cols:
        return
    names = [c.split("::", 1)[1] for c in feat_cols]
    matrix = gf[feat_cols].to_numpy().T  # features x generations

    plt, fig, ax = _matplotlib_axes((max(8.0, len(gf) * 0.3), max(6.0, len(names) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xticks(range(len(gf)))
    ax.set_xticklabels(gf["generation"], fontsize=7)
    ax.set_xlabel("generation")
    ax.set_title("feature selection frequency by generation")
    fig.colorbar(im, ax=ax, label="share of population with feature on")
    fig.tight_layout()
    fig.savefig(run_dir / "feature_frequency_heatmap.png", dpi=120)
    plt.close(fig)


def render_all(run_dir: str | Path) -> None:
    """(Re)renders `champion_family_tree.txt` and the three PNGs from
    whatever is currently on disk in `run_dir`. Idempotent; safe to call
    repeatedly (once at the end of `evolve()`, again later via `evo-report`
    once the controls have finished on a subsequent night).
    """
    run_dir = Path(run_dir)
    gen_csv = run_dir / "generations.csv"
    if not gen_csv.exists():
        print(f"[outputs] {gen_csv} not found -- nothing to render yet")
        return

    gens = pd.read_csv(gen_csv)
    _render_fitness_plot(run_dir, gens)
    _render_diversity_plot(run_dir, gens)

    gf_csv = run_dir / "gene_frequency.csv"
    if gf_csv.exists():
        _render_feature_heatmap(run_dir, pd.read_csv(gf_csv))

    progress_path = run_dir / "progress.json"
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
        champion_id = progress.get("champion_id")
        if champion_id:
            from stockml.evolution.lineage import render_family_tree

            try:
                text = render_family_tree(run_dir, champion_id)
                with open(run_dir / "champion_family_tree.txt", "w", encoding="utf-8") as f:
                    f.write(text)
            except KeyError as e:
                print(f"[outputs] could not render family tree: {e}")

    missing = []
    if not (run_dir / "control_random" / "summary.json").exists():
        missing.append("control_random")
    if not (run_dir / "control_null" / "progress.json").exists():
        missing.append("control_null")
    if not (run_dir / "vault_report.txt").exists():
        missing.append("vault")
    if missing:
        print(f"[outputs] not yet run for this run: {', '.join(missing)}")
    print(f"[outputs] rendered plots + family tree -> {run_dir}")
