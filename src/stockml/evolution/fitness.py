"""A genome's fitness: fit it on its own training window, predict a zone
walk-forward exactly as step 1 does, and score its day-level edge against
the run's one canonical majority-class baseline (CLAUDE.md: "a genome may
choose what it looks at and how it learns, never what it is judged on").

Performance note (see CLAUDE.md): the block-bootstrap CI this computes for
*every* genome uses a small `n_boot` by default -- the hot loop evaluates on
the order of a thousand genomes, and step 1's `n_boot=2000` per genome would
be real, avoidable wall-clock cost. The vault protocol and final champion
reporting re-evaluate the short final list of individuals with a much larger
`n_boot` for the numbers that actually get written up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from stockml import evaluate as eval_mod
from stockml.evolution.genome import Genome
from stockml.evolution.model_builder import build_model
from stockml.evolution.zones import EvoZones
from stockml.features import feature_names
from stockml.models.baselines import MajorityClass
from stockml.update import Frozen
from stockml.walk_forward import walk_forward_single_model

DEFAULT_FITNESS_N_BOOT = 200


@dataclass(frozen=True)
class FitnessResult:
    genome_hash: str
    zone: str
    n_days: int
    mean_edge_pp: float
    se_pp: float
    fitness: float  # mean_edge_pp - se_pp -- the selection scalar
    accuracy: float
    ci95_low_pp: float
    ci95_high_pp: float
    prediction_mix: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_hash": self.genome_hash,
            "zone": self.zone,
            "n_days": self.n_days,
            "mean_edge_pp": self.mean_edge_pp,
            "se_pp": self.se_pp,
            "fitness": self.fitness,
            "accuracy": self.accuracy,
            "ci95_low_pp": self.ci95_low_pp,
            "ci95_high_pp": self.ci95_high_pp,
            "prediction_mix": self.prediction_mix,
        }


def _training_window(genome: Genome, zones: EvoZones) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = zones.train_end - pd.DateOffset(years=genome.train_years_used)
    start = max(start, zones.train_start)
    return start, zones.train_end


def evaluate_genome(
    genome: Genome,
    zone: str,
    dataset: pd.DataFrame,
    zones: EvoZones,
    majority_daily: pd.Series,
    seed: int,
    n_boot: int = DEFAULT_FITNESS_N_BOOT,
) -> FitnessResult:
    """Fit `genome`'s model on its own training window, predict `zone`
    ("arena" or "vault") walk-forward (frozen, exactly as step 1), and
    return its day-level edge against `majority_daily` (the run's one
    canonical majority-baseline daily-accuracy series for this zone).
    """
    if zone not in ("arena", "vault"):
        raise ValueError(f"zone must be 'arena' or 'vault', got {zone!r}")

    dates = pd.to_datetime(dataset["date"])
    train_start, train_end = _training_window(genome, zones)
    train_df = dataset[(dates >= train_start) & (dates <= train_end)]

    if zone == "arena":
        z0, z1 = zones.arena_start, zones.arena_end
    else:
        z0, z1 = zones.vault_start, zones.vault_end
    eval_df = dataset[(dates >= z0) & (dates <= z1)]

    active = genome.active_features()
    model = build_model(genome, seed)
    preds, _, _, _ = walk_forward_single_model(
        train_df, eval_df, eval_df.iloc[0:0], model, Frozen(), active
    )

    model_daily = eval_mod.daily_accuracy(preds["date"], preds["y_true"], preds["y_pred"])
    edge = (model_daily - majority_daily).dropna()

    accuracy = float((preds["y_true"] == preds["y_pred"]).mean()) if len(preds) else 0.0
    mix = eval_mod.prediction_mix(preds["y_pred"])

    n_days = len(edge)
    if n_days == 0:
        mean_edge_pp = se_pp = 0.0
        fitness = 0.0
        ci_low = ci_high = 0.0
    else:
        edge_arr = edge.to_numpy()
        mean_edge = float(edge_arr.mean())
        se = float(edge_arr.std(ddof=1) / np.sqrt(n_days)) if n_days > 1 else 0.0
        mean_edge_pp = mean_edge * 100
        se_pp = se * 100
        fitness = mean_edge_pp - se_pp
        if n_days > 1 and n_boot > 0:
            ci_low, ci_high = eval_mod.block_bootstrap_ci(edge_arr, block_size=20, n_boot=n_boot, seed=seed)
            ci_low, ci_high = ci_low * 100, ci_high * 100
        else:
            ci_low = ci_high = mean_edge_pp

    return FitnessResult(
        genome_hash=genome.stable_hash(),
        zone=zone,
        n_days=n_days,
        mean_edge_pp=mean_edge_pp,
        se_pp=se_pp,
        fitness=fitness,
        accuracy=accuracy,
        ci95_low_pp=ci_low,
        ci95_high_pp=ci_high,
        prediction_mix=mix,
    )


def canonical_majority_daily(dataset: pd.DataFrame, zones: EvoZones, zone: str) -> pd.Series:
    """The run's one majority-class baseline, trained on the full/global
    train window (independent of any genome's own `train_years_used`) and
    predicted on `zone` -- every genome's fitness is measured against this
    same series ("never what it is judged on").
    """
    dates = pd.to_datetime(dataset["date"])
    train_df = dataset[(dates >= zones.train_start) & (dates <= zones.train_end)]
    if zone == "arena":
        z0, z1 = zones.arena_start, zones.arena_end
    else:
        z0, z1 = zones.vault_start, zones.vault_end
    eval_df = dataset[(dates >= z0) & (dates <= z1)]

    model = MajorityClass()
    preds, _, _, _ = walk_forward_single_model(
        train_df, eval_df, eval_df.iloc[0:0], model, Frozen(), feature_names()
    )
    return eval_mod.daily_accuracy(preds["date"], preds["y_true"], preds["y_pred"])
