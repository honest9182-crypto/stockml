"""Full lineage: every individual ever created is written to
`lineage.jsonl`, one JSON object per line, and traceable back to
generation 0 by `parents`. This is what makes "any champion must be
traceable back to generation 0" (CLAUDE.md) an enforced fact, not a hope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stockml.evolution.population import Individual


def individual_to_record(ind: Individual) -> dict[str, Any]:
    return {
        "id": ind.id,
        "generation": ind.generation,
        "genome": ind.genome.encode(),
        "genome_hash": ind.genome.stable_hash(),
        "parents": ind.parents,
        "origin": ind.origin,
        "mutated_genes": ind.mutated_genes,
        "storm": ind.storm,
        "fitness": ind.fitness.to_dict() if ind.fitness is not None else None,
        "fitness_ref": ind.fitness_ref,
        "memoized": ind.memoized,
    }


def append_lineage(run_dir: str | Path, individuals: list[Individual]) -> None:
    path = Path(run_dir) / "lineage.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        for ind in individuals:
            f.write(json.dumps(individual_to_record(ind), sort_keys=True) + "\n")


def read_lineage(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """All individuals ever recorded in this run, keyed by id."""
    path = Path(run_dir) / "lineage.jsonl"
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["id"]] = rec
    return records


def trace_ancestors(run_dir: str | Path, individual_id: str) -> list[dict[str, Any]]:
    """Every ancestor of `individual_id` (including itself), reachable by
    walking `parents` back to an empty list, ordered oldest generation
    first. Raises `KeyError` if `individual_id` isn't in this run's lineage.
    """
    records = read_lineage(run_dir)
    if individual_id not in records:
        raise KeyError(f"individual {individual_id!r} not found in {run_dir}/lineage.jsonl")

    visited: dict[str, dict[str, Any]] = {}

    def visit(iid: str) -> None:
        if iid in visited:
            return
        rec = records.get(iid)
        if rec is None:
            return  # a parent id not present in this run's lineage -- skip, don't crash
        visited[iid] = rec
        for p in rec["parents"]:
            visit(p)

    visit(individual_id)
    return sorted(visited.values(), key=lambda r: (r["generation"], r["id"]))


def render_family_tree(run_dir: str | Path, individual_id: str) -> str:
    """Human-readable ancestor chain, generation 0 first, annotated with the
    mutation (if any) that produced each individual.
    """
    ancestors = trace_ancestors(run_dir, individual_id)
    lines = [f"Family tree of {individual_id}:", ""]
    for rec in ancestors:
        indent = "  " * rec["generation"]
        bits = [f"gen{rec['generation']:02d}", rec["id"], f"origin={rec['origin']}"]
        if rec["parents"]:
            bits.append(f"parents={rec['parents']}")
        if rec["mutated_genes"]:
            bits.append(f"mutated={rec['mutated_genes']}")
        if rec["storm"]:
            bits.append("[STORM]")
        if rec["fitness"] is not None:
            bits.append(f"fitness={rec['fitness']['fitness']:.4f}pp")
        lines.append(indent + " ".join(bits))
    return "\n".join(lines)
