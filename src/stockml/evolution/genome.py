"""Genome v1: a frozen, hashable description of one model -- which features
it looks at, which model family, and that family's hyperparameters.

Design rule (per CLAUDE.md): every field has a default, so a `Genome` saved
to a `lineage.jsonl` by an older version of this file still `decode()`s
correctly after new genes are added -- unknown keys in a saved dict are
ignored, missing keys fall back to the field default.

Genes that don't apply to the genome's current `model_family` (e.g. the
`hgb_*` genes on a `model_family="logreg"` genome) are still stored,
mutated, and inherited like any other gene -- "recessive genes" that can
resurface if a child's `model_family` flips. `model_builder.py` is the only
place that decides which genes actually matter for a given genome.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from stockml.features import feature_names

FEATURE_NAMES: list[str] = feature_names()
N_FEATURES: int = len(FEATURE_NAMES)
MIN_FEATURES_ON: int = 2

# Ordered so "a value between the two parents' grid positions" (crossover)
# and "step one position up or down" (mutation) are well-defined. The two
# categorical genes (no natural order) still get an order here -- crossover
# and mutation treat them as a 2-element grid, which degenerates cleanly to
# "pick either parent's value" / "resample", exactly as it should.
GENE_GRIDS: dict[str, tuple[Any, ...]] = {
    "model_family": ("logreg", "hgb"),
    "train_years_used": (1, 2, 3),
    "stagnant_bias": (-0.10, -0.05, 0.0, 0.05, 0.10),
    "lr_C": (0.01, 0.1, 1.0, 10.0),
    "class_weight": ("none", "balanced"),
    "hgb_max_depth": (None, 2, 3, 4, 6),
    "hgb_learning_rate": (0.03, 0.1, 0.3),
    "hgb_max_iter": (50, 100, 200, 400),
    "hgb_min_samples_leaf": (20, 100, 500, 2000),
    "hgb_l2": (0.0, 1.0, 10.0),
}

# Genes with a natural numeric order -- mutation "steps" these one grid
# position; the two categorical genes (no order) get resampled instead.
CATEGORICAL_GENES = frozenset({"model_family", "class_weight"})
ORDERED_GENES = frozenset(GENE_GRIDS) - CATEGORICAL_GENES


@dataclass(frozen=True)
class Genome:
    feature_mask: tuple[bool, ...] = (True,) * N_FEATURES
    model_family: str = "hgb"
    train_years_used: int = 3
    stagnant_bias: float = 0.0
    lr_C: float = 1.0
    class_weight: str = "none"
    hgb_max_depth: int | None = None
    hgb_learning_rate: float = 0.1
    hgb_max_iter: int = 100
    hgb_min_samples_leaf: int = 20
    hgb_l2: float = 0.0

    def __post_init__(self) -> None:
        if len(self.feature_mask) != N_FEATURES:
            raise ValueError(f"feature_mask has {len(self.feature_mask)} bits, expected {N_FEATURES}")
        if sum(self.feature_mask) < MIN_FEATURES_ON:
            raise ValueError(f"feature_mask has < {MIN_FEATURES_ON} features on")
        for name, grid in GENE_GRIDS.items():
            if getattr(self, name) not in grid:
                raise ValueError(f"{name}={getattr(self, name)!r} not in its grid {grid}")

    def active_features(self) -> list[str]:
        return [name for name, on in zip(FEATURE_NAMES, self.feature_mask) if on]

    def n_features_on(self) -> int:
        return sum(self.feature_mask)

    def encode(self) -> dict[str, Any]:
        """Flat, JSON-safe dict -- feature_mask as a '0'/'1' string."""
        d = asdict(self)
        d["feature_mask"] = "".join("1" if b else "0" for b in self.feature_mask)
        return d

    @classmethod
    def decode(cls, d: dict[str, Any]) -> "Genome":
        """Inverse of `encode`. Unknown keys ignored; missing keys use the
        field's default, so genomes saved before a gene existed still decode.
        """
        kwargs: dict[str, Any] = {}
        if "feature_mask" in d:
            kwargs["feature_mask"] = tuple(c == "1" for c in d["feature_mask"])
        for f in fields(cls):
            if f.name == "feature_mask":
                continue
            if f.name in d:
                kwargs[f.name] = d[f.name]
        return cls(**kwargs)

    def stable_hash(self) -> str:
        """Deterministic across processes/Python versions -- unlike the
        built-in `hash()`, which is randomized per-process for str fields
        (PYTHONHASHSEED) and therefore useless as a cross-run memoization
        key or a reproducible individual id.
        """
        canonical = json.dumps(self.encode(), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# The two seeded individuals: sklearn's own defaults, all features on --
# these reproduce step 1's LogReg and HGB exactly, so a report can always
# show whether evolution ever beat where the search started.
SEED_LOGREG = Genome(model_family="logreg")
SEED_HGB = Genome(model_family="hgb")


def _ensure_min_features(mask: tuple[bool, ...], rng: np.random.Generator) -> tuple[bool, ...]:
    mask_list = list(mask)
    on = [i for i, b in enumerate(mask_list) if b]
    off = [i for i, b in enumerate(mask_list) if not b]
    while len(on) < MIN_FEATURES_ON:
        i = off.pop(rng.integers(0, len(off)))
        mask_list[i] = True
        on.append(i)
    return tuple(mask_list)


def random_genome(rng: np.random.Generator) -> Genome:
    """Bernoulli(0.5) feature bits (re-filled up to the minimum), uniform
    draw from each gene's grid.
    """
    mask = tuple(bool(x) for x in rng.integers(0, 2, size=N_FEATURES))
    mask = _ensure_min_features(mask, rng)
    kwargs: dict[str, Any] = {"feature_mask": mask}
    for name, grid in GENE_GRIDS.items():
        kwargs[name] = grid[rng.integers(0, len(grid))]
    return Genome(**kwargs)


def mutate(genome: Genome, rng: np.random.Generator, rate: float) -> tuple[Genome, list[str]]:
    """Each feature bit flips with probability `rate`; each grid gene
    mutates with probability `rate` (ordered genes step one grid position,
    categorical genes resample uniformly). Re-validates the minimum active
    feature count. Returns (child, names of genes that actually changed
    value -- a flipped bit that happened to already match nothing, or a
    step/resample that lands back on the same value, is not "mutated").
    """
    mutated: list[str] = []
    mask = list(genome.feature_mask)
    for i in range(N_FEATURES):
        if rng.random() < rate:
            mask[i] = not mask[i]
            mutated.append(f"feature_bit_{i}:{FEATURE_NAMES[i]}")
    fixed_mask = _ensure_min_features(tuple(mask), rng)
    if fixed_mask != tuple(mask):
        mutated.append("feature_mask_min_repair")
    mask = fixed_mask

    kwargs: dict[str, Any] = {"feature_mask": mask}
    for name, grid in GENE_GRIDS.items():
        current = getattr(genome, name)
        if rng.random() < rate:
            if name in ORDERED_GENES:
                idx = grid.index(current)
                direction = 1 if rng.random() < 0.5 else -1
                new_idx = min(max(idx + direction, 0), len(grid) - 1)
                new_val = grid[new_idx]
            else:
                new_val = grid[rng.integers(0, len(grid))]
            if new_val != current:
                mutated.append(name)
            kwargs[name] = new_val
        else:
            kwargs[name] = current
    return Genome(**kwargs), mutated


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    """Uniform crossover: each feature bit and each grid gene independently
    from either parent (p=0.5). For grid genes, with p=0.25 the child
    instead gets a value drawn uniformly from the grid positions between the
    two parents' values (inclusive) -- for the two categorical genes (no
    real "between") this degenerates to picking either parent's value,
    which is harmless.
    """
    mask = tuple(
        a.feature_mask[i] if rng.random() < 0.5 else b.feature_mask[i] for i in range(N_FEATURES)
    )
    mask = _ensure_min_features(mask, rng)

    kwargs: dict[str, Any] = {"feature_mask": mask}
    for name, grid in GENE_GRIDS.items():
        va, vb = getattr(a, name), getattr(b, name)
        if rng.random() < 0.25:
            ia, ib = grid.index(va), grid.index(vb)
            lo, hi = min(ia, ib), max(ia, ib)
            kwargs[name] = grid[rng.integers(lo, hi + 1)]
        else:
            kwargs[name] = va if rng.random() < 0.5 else vb
    return Genome(**kwargs)


def genetic_distance(a: Genome, b: Genome) -> float:
    """Fraction of "gene slots" that differ: each feature bit counts as its
    own slot (Hamming distance on the mask), each grid gene counts as one
    slot (differs or not, not weighted by how far apart on the grid). In
    [0, 1].
    """
    n_diff_bits = sum(1 for x, y in zip(a.feature_mask, b.feature_mask) if x != y)
    n_diff_genes = sum(1 for name in GENE_GRIDS if getattr(a, name) != getattr(b, name))
    total_slots = N_FEATURES + len(GENE_GRIDS)
    return (n_diff_bits + n_diff_genes) / total_slots
