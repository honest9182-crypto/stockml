"""The vault protocol (CLAUDE.md): opened exactly once, for a fixed,
pre-declared list of individuals, at the end of a run -- never during. The
guard here is filesystem state (`progress.json`'s `status`), not a runtime
flag a caller could bypass: `run_vault` refuses to open a run that's still
`"running"`.

A vault result is never a reason to change the config and run again. Every
look is permanently appended to the global `runs/evolution/vault_log.jsonl`
-- nothing is ever removed from it, including after a later, "better" run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from stockml import evaluate as eval_mod
from stockml import run as run_mod
from stockml.evolution import lineage as lineage_mod
from stockml.evolution.fitness import FitnessResult, canonical_majority_daily, evaluate_genome
from stockml.evolution.genome import SEED_HGB, SEED_LOGREG, Genome
from stockml.evolution.loop import build_evo_dataset, read_progress
from stockml.evolution.zones import EvoZones, compute_evo_zones
from stockml.features import feature_names
from stockml.models.baselines import MajorityClass
from stockml.update import Frozen
from stockml.walk_forward import walk_forward_single_model

GLOBAL_VAULT_LOG = Path("runs/evolution/vault_log.jsonl")


class VaultGuardError(RuntimeError):
    """Raised when the vault is opened for a run that's still running."""


def _assert_not_running(run_dir: Path) -> dict[str, Any]:
    progress = read_progress(run_dir)
    if progress["status"] == "running":
        raise VaultGuardError(
            f"{run_dir} is still running (progress.json status='running') -- "
            f"the vault cannot be opened while an evolution is in progress"
        )
    return progress


def _best_genomes_by_hash(records: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """(genome_hash, best-fitness record seen for it), sorted best first."""
    best: dict[str, dict[str, Any]] = {}
    for rec in records.values():
        if rec["fitness"] is None:
            continue
        h = rec["genome_hash"]
        if h not in best or rec["fitness"]["fitness"] > best[h]["fitness"]["fitness"]:
            best[h] = rec
    return sorted(best.items(), key=lambda kv: kv[1]["fitness"]["fitness"], reverse=True)


def build_vault_list(run_dir: Path) -> dict[str, Genome | None]:
    """The fixed, pre-declared list of individuals the vault opens for:
    the champion, the top 5 by arena fitness, the two step-1 seeds, the
    random-search control's champion, the null control's champion, and the
    majority baseline (`None` -- not a genome, handled specially).
    """
    records = lineage_mod.read_lineage(run_dir)
    ranked = _best_genomes_by_hash(records)
    if not ranked:
        raise RuntimeError(f"no scored individuals found in {run_dir}/lineage.jsonl")

    vault_list: dict[str, Genome | None] = {"champion": Genome.decode(ranked[0][1]["genome"])}
    for i, (_, rec) in enumerate(ranked[:5], start=1):
        vault_list[f"top{i}"] = Genome.decode(rec["genome"])
    vault_list["seed_logreg"] = SEED_LOGREG
    vault_list["seed_hgb"] = SEED_HGB

    random_summary_path = run_dir / "control_random" / "summary.json"
    if random_summary_path.exists():
        with open(random_summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        vault_list["random_search_champion"] = Genome.decode(summary["champion_genome"])
    else:
        print(f"[vault] {random_summary_path} not found -- random-search control skipped in this vault look")

    null_ranked = _best_genomes_by_hash(lineage_mod.read_lineage(run_dir / "control_null"))
    if null_ranked:
        vault_list["null_champion"] = Genome.decode(null_ranked[0][1]["genome"])
    else:
        print(f"[vault] {run_dir}/control_null/lineage.jsonl not found -- null control skipped in this vault look")

    vault_list["majority_baseline"] = None
    return vault_list


def _evaluate_majority(dataset: pd.DataFrame, zones: EvoZones, zone: str) -> FitnessResult:
    """The majority baseline's own accuracy in `zone` -- not "edge vs
    itself" (which is trivially zero), reported for context alongside every
    genome's edge against this same baseline.
    """
    dates = pd.to_datetime(dataset["date"])
    train_df = dataset[(dates >= zones.train_start) & (dates <= zones.train_end)]
    z0, z1 = (zones.arena_start, zones.arena_end) if zone == "arena" else (zones.vault_start, zones.vault_end)
    eval_df = dataset[(dates >= z0) & (dates <= z1)]
    preds, _, _, _ = walk_forward_single_model(
        train_df, eval_df, eval_df.iloc[0:0], MajorityClass(), Frozen(), feature_names()
    )
    accuracy = float((preds["y_true"] == preds["y_pred"]).mean()) if len(preds) else 0.0
    mix = eval_mod.prediction_mix(preds["y_pred"]) if len(preds) else {"down": 0.0, "stagnant": 0.0, "up": 0.0}
    n_days = len(eval_mod.daily_accuracy(preds["date"], preds["y_true"], preds["y_pred"])) if len(preds) else 0
    return FitnessResult(
        genome_hash="majority_baseline", zone=zone, n_days=n_days,
        mean_edge_pp=0.0, se_pp=0.0, fitness=0.0, accuracy=accuracy,
        ci95_low_pp=0.0, ci95_high_pp=0.0, prediction_mix=mix,
    )


def _render_vault_report(run_dir: Path, results: dict[str, Any]) -> str:
    lines = [
        f"=== vault report: {run_dir} ===",
        "",
        "A vault result is never a reason to change the config and run again.",
        "Every look here is permanently logged in runs/evolution/vault_log.jsonl.",
        "",
        f"{'entry':<24}{'arena_fitness':>14}{'vault_fitness':>14}{'vault_CI_low':>14}"
        f"{'vault_CI_high':>14}{'vault_accuracy':>15}",
    ]
    lines.append("-" * len(lines[-1]))
    for label, r in results.items():
        a, v = r["arena"], r["vault"]
        lines.append(
            f"{label:<24}{a['fitness']:>14.4f}{v['fitness']:>14.4f}{v['ci95_low_pp']:>14.4f}"
            f"{v['ci95_high_pp']:>14.4f}{v['accuracy']:>15.4f}"
        )
    lines.append("")
    lines.append(
        "Expected honest picture: champion's arena fitness > the seeds', and its vault edge"
    )
    lines.append(
        "shrinks toward zero. How much it shrinks -- and whether it clears the null's vault"
    )
    lines.append("fitness (the luck ceiling) -- is the finding.")
    return "\n".join(lines)


def _append_vault_log(run_dir: Path, cfg: dict[str, Any], results: dict[str, Any]) -> None:
    GLOBAL_VAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_id": run_dir.name,
        "config_name": cfg.get("name"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    with open(GLOBAL_VAULT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def run_vault(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    _assert_not_running(run_dir)
    with open(run_dir / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_mod.apply_env_overrides(cfg)

    vault_list = build_vault_list(run_dir)

    raw_panel, dataset = build_evo_dataset(cfg)
    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(dataset["date"]).unique()))
    zones = compute_evo_zones(
        unique_dates, cfg["split"]["train_years"], cfg["split"]["sanity_days"],
        cfg["split"].get("embargo_days", 1), cfg["zones"]["arena_end"], cfg["zones"]["vault_start"],
    )
    majority_arena = canonical_majority_daily(dataset, zones, "arena")
    majority_vault = canonical_majority_daily(dataset, zones, "vault")
    n_boot = cfg["evolution"]["report_n_boot"]

    results: dict[str, Any] = {}
    for label, genome in vault_list.items():
        if genome is None:
            arena_r = _evaluate_majority(dataset, zones, "arena")
            vault_r = _evaluate_majority(dataset, zones, "vault")
        else:
            arena_r = evaluate_genome(genome, "arena", dataset, zones, majority_arena, cfg["seed"], n_boot)
            vault_r = evaluate_genome(genome, "vault", dataset, zones, majority_vault, cfg["seed"], n_boot)
        results[label] = {
            "genome": genome.encode() if genome is not None else None,
            "arena": arena_r.to_dict(),
            "vault": vault_r.to_dict(),
        }
        print(
            f"[vault] {label}: arena={arena_r.fitness:+.4f}pp vault={vault_r.fitness:+.4f}pp "
            f"(95% CI [{vault_r.ci95_low_pp:+.3f}, {vault_r.ci95_high_pp:+.3f}]pp)"
        )

    report_text = _render_vault_report(run_dir, results)
    with open(run_dir / "vault_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)

    _append_vault_log(run_dir, cfg, results)
    print(f"[vault] appended this look to {GLOBAL_VAULT_LOG}")
    return run_dir
