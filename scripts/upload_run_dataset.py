#!/usr/bin/env python
"""Package a completed `runs/<name>/` folder as a private Kaggle Dataset,
via the `kaggle` CLI, so `kaggle/stage.ipynb`'s `resume`/`random`/`null`
stages can attach it as `PREV_RUN_INPUT_SLUG` without the manual "Add Input
-> Notebook Output" step (which only exists in Kaggle's web UI, not the
API/CLI this repo otherwise drives everything through).

Staged with the run folder nested under `runs/<run-name>/` (not flat) --
`stage.ipynb`'s "Bring in a previous run" cell looks for
`<mounted-dataset>/runs/<RUN_NAME>`, matching exactly how a run's own
`STOCKML_RUNS_DIR=/kaggle/working/runs` lays it out, so this dataset's
layout has to mirror that, not `data/cache`'s flat one
(`upload_cache_dataset.py`'s different convention, for a different mount
point).

Requires the `kaggle` CLI (`pip install kaggle`) with credentials
configured -- see `upload_cache_dataset.py`'s docstring for the same
authentication and safety notes (private by default, never touches your
account until invoked, and the same benign kaggle-CLI-exits-non-zero-after-
succeeding quirk documented there applies here too).

Usage:
    # one-time: create the dataset (private by default)
    python scripts/upload_run_dataset.py runs/evo_<ts>_<name> --kaggle-username YOURNAME

    # later: push a newer/different run under the same dataset slug
    python scripts/upload_run_dataset.py runs/evo_<ts>_<name> --kaggle-username YOURNAME --version

    # inspect what would be staged/run without touching Kaggle at all
    python scripts/upload_run_dataset.py runs/evo_<ts>_<name> --kaggle-username YOURNAME --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STAGING_DIR = REPO_ROOT / "kaggle" / ".run_dataset_staging"


def _kaggle_cmd_prefix() -> list[str]:
    """Same reasoning as upload_cache_dataset.py's helper of the same name:
    prefer `kaggle` on PATH, else fall back to `python -m kaggle` under
    this interpreter.
    """
    found = shutil.which("kaggle")
    return [found] if found else [sys.executable, "-m", "kaggle"]


def _detect_kaggle_username() -> str | None:
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        return None
    try:
        with open(cred_path, "r", encoding="utf-8") as f:
            return json.load(f).get("username")
    except Exception:
        return None


def build_staging_dir(
    run_dir: Path,
    staging_dir: Path,
    title: str,
    dataset_id: str,
    license_name: str,
) -> Path:
    """Materializes `staging_dir` as `<staging_dir>/runs/<run_dir.name>/...`
    (a full copy of `run_dir`) + `dataset-metadata.json` at the root.
    """
    if not run_dir.is_dir():
        raise SystemExit(f"run dir {run_dir} does not exist")
    if not (run_dir / "progress.json").exists():
        raise SystemExit(f"{run_dir} has no progress.json -- not a run directory?")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    dest = staging_dir / "runs" / run_dir.name
    shutil.copytree(run_dir, dest)

    metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": license_name}],
    }
    with open(staging_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return staging_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path, help="Path to the run directory, e.g. runs/evo_<ts>_<name>")
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument(
        "--kaggle-username", default=None,
        help="Your Kaggle username -- the dataset id is '<username>/<slug>'. "
             "Auto-detected from ~/.kaggle/kaggle.json if omitted.",
    )
    parser.add_argument(
        "--slug", default=None,
        help="Dataset slug (URL-safe, lowercase, hyphenated). Default: 'stockml-run-<run-dir-name-with-underscores-as-hyphens>'.",
    )
    parser.add_argument("--title", default=None, help="Human-readable dataset title. Default: derived from the slug.")
    parser.add_argument("--license", default="CC0-1.0", dest="license_name", help="Kaggle license short name.")
    parser.add_argument(
        "--version", action="store_true",
        help="Push a new version of an existing dataset (kaggle datasets version) instead of creating one.",
    )
    parser.add_argument(
        "--version-notes", default="Updated run.",
        help="Version notes, only used with --version.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build the staging directory and print the kaggle command, but don't run it.",
    )
    args = parser.parse_args()

    username = args.kaggle_username or _detect_kaggle_username()
    if not username:
        raise SystemExit(
            "no Kaggle username: pass --kaggle-username, or set one up via "
            "~/.kaggle/kaggle.json (see https://www.kaggle.com/docs/api#authentication)"
        )
    slug = args.slug or f"stockml-run-{args.run_dir.name.replace('_', '-')}"
    title = args.title or slug.replace("-", " ")
    dataset_id = f"{username}/{slug}"

    staging_dir = build_staging_dir(args.run_dir, args.staging_dir, title, dataset_id, args.license_name)
    print(f"[upload_run_dataset] staged {args.run_dir} -> {staging_dir}/runs/{args.run_dir.name}")
    print(f"[upload_run_dataset] dataset id: {dataset_id}")

    kaggle_prefix = _kaggle_cmd_prefix()
    # -t/--keep-tabular: see upload_cache_dataset.py's identical note.
    # -r zip: unlike upload_cache_dataset.py's deliberately flat staging,
    # this dataset's whole point is the nested runs/<name>/ folder -- the
    # kaggle CLI otherwise silently skips subdirectories ("Skipping folder:
    # runs; use '--dir-mode' to upload folders", observed directly). Kaggle
    # extracts the zip server-side; the mounted dataset ends up with the
    # real runs/<name>/ directory tree, not a .zip file to unpack.
    common_flags = ["-t", "-r", "zip"]
    if args.version:
        cmd = kaggle_prefix + ["datasets", "version", "-p", str(staging_dir), "-m", args.version_notes] + common_flags
    else:
        cmd = kaggle_prefix + ["datasets", "create", "-p", str(staging_dir)] + common_flags
        # `kaggle datasets create` is private by default -- this never adds
        # -u/--public, on purpose.

    print(f"[upload_run_dataset] command: {' '.join(cmd)}")
    if args.dry_run:
        print("[upload_run_dataset] --dry-run: not invoking kaggle. Review the staged files above, "
              "then re-run without --dry-run (or copy/paste the command).")
        return

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # See upload_cache_dataset.py's identical note: the kaggle CLI can
        # exit non-zero on a final housekeeping call even after a fully
        # successful upload. Verify before retrying.
        print(
            f"\n[upload_run_dataset] kaggle exited non-zero. This can happen even when the "
            f"upload actually succeeded -- verify before retrying:\n"
            f"  kaggle datasets files {dataset_id}\n"
        )
        raise


if __name__ == "__main__":
    main()
