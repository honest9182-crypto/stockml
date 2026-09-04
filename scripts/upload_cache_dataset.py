#!/usr/bin/env python
"""Package `data/cache/*.parquet` and `data/tickers/sp500.csv` as a private
Kaggle Dataset, via the `kaggle` CLI.

Kaggle notebooks are the cheapest way to run the overnight
`configs/evo.yaml` search (see README's "Running on Kaggle"), but a
Kaggle notebook has no access to your local `data/cache/`. This script
stages a flat copy of the cache + a pinned snapshot of the ticker universe
and shells out to `kaggle datasets create` (first upload) or
`kaggle datasets version` (`--version`, to refresh an existing dataset as
your local cache grows -- re-run `stockml download` locally, then this,
whenever the cache gets meaningfully wider).

The staging directory is deliberately flat (no subfolders): `stage.ipynb`
symlinks the whole dataset onto `data/cache`, so its files need to look
exactly like `data/cache`'s own contents (`<TICKER>.parquet`, one per
ticker) for `load_panel`/`download_prices` to find them; `sp500.csv` rides
along flat too, landing inside the symlinked `data/cache` as harmless
clutter nothing reads from there -- it's included only because the repo's
own `data/tickers/sp500.csv` can drift after the cache was built, and this
gives you a pinned copy of the universe that produced this exact cache.

Requires the `kaggle` CLI (`pip install kaggle`) with credentials
configured (`~/.kaggle/kaggle.json` holding your API token -- see
https://www.kaggle.com/docs/api#authentication). This script only builds
the staging directory and prints/runs the `kaggle` command; it never
touches your Kaggle account unless you actually invoke it (or a fresh
`kaggle datasets create`/`version` call it makes fails on its own, e.g. for
bad credentials -- nothing here can accidentally make the dataset public:
`kaggle datasets create` is private by default and this script never
passes `-u`/`--public`).

Usage:
    # one-time: create the dataset (private by default)
    python scripts/upload_cache_dataset.py --kaggle-username YOURNAME

    # later: push a new version once the local cache has grown
    python scripts/upload_cache_dataset.py --kaggle-username YOURNAME --version

    # inspect what would be staged/run without touching Kaggle at all
    python scripts/upload_cache_dataset.py --kaggle-username YOURNAME --dry-run

Known `kaggle` CLI quirk, observed directly uploading the full ~500-ticker
universe: it can print "Upload successful" for every single file and then
exit non-zero on a JSON-decode error in some final housekeeping call, even
though the dataset was actually created/updated correctly. This script's
own error message on that path tells you the exact `kaggle datasets files
<id>` command to check before assuming the upload didn't happen.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _kaggle_cmd_prefix() -> list[str]:
    """The `kaggle` console script isn't guaranteed to be on PATH (e.g. it's
    only on a different Python installation's Scripts/bin dir than the one
    running this script) -- prefer it when it is (`shutil.which`), otherwise
    fall back to `python -m kaggle` under *this* interpreter, which works
    as long as `kaggle` is installed wherever this script is being run
    from (`pip install kaggle`).
    """
    found = shutil.which("kaggle")
    return [found] if found else [sys.executable, "-m", "kaggle"]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"
DEFAULT_TICKERS_CSV = REPO_ROOT / "data" / "tickers" / "sp500.csv"
DEFAULT_STAGING_DIR = REPO_ROOT / "kaggle" / ".dataset_staging"


def _detect_kaggle_username() -> str | None:
    """Best-effort read of the `username` field kaggle.json already has --
    saves typing --kaggle-username every time once you're authenticated."""
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        return None
    try:
        with open(cred_path, "r", encoding="utf-8") as f:
            return json.load(f).get("username")
    except Exception:
        return None


def build_staging_dir(
    cache_dir: Path,
    tickers_csv: Path,
    staging_dir: Path,
    title: str,
    dataset_id: str,
    license_name: str,
) -> Path:
    """Materializes `staging_dir`: dataset-metadata.json + a flat copy of
    every cached ticker's parquet (no .meta.json sidecars -- only
    `data/cache/*.parquet`, as asked) + `sp500.csv`.
    """
    if not cache_dir.is_dir():
        raise SystemExit(f"cache dir {cache_dir} does not exist -- run `stockml download` first")
    parquet_files = sorted(cache_dir.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"no *.parquet files found under {cache_dir} -- run `stockml download` first")
    if not tickers_csv.exists():
        raise SystemExit(f"tickers csv {tickers_csv} does not exist")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    for p in parquet_files:
        shutil.copy2(p, staging_dir / p.name)
    shutil.copy2(tickers_csv, staging_dir / tickers_csv.name)

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
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--tickers-csv", type=Path, default=DEFAULT_TICKERS_CSV)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument(
        "--kaggle-username", default=None,
        help="Your Kaggle username -- the dataset id is '<username>/<slug>'. "
             "Auto-detected from ~/.kaggle/kaggle.json if omitted.",
    )
    parser.add_argument("--slug", default="stockml-price-cache", help="Dataset slug (URL-safe, lowercase, hyphenated).")
    parser.add_argument("--title", default="stockml price cache", help="Human-readable dataset title.")
    parser.add_argument("--license", default="CC0-1.0", dest="license_name", help="Kaggle license short name.")
    parser.add_argument(
        "--version", action="store_true",
        help="Push a new version of an existing dataset (kaggle datasets version) instead of creating one.",
    )
    parser.add_argument(
        "--version-notes", default="Updated price cache.",
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
    dataset_id = f"{username}/{args.slug}"

    staging_dir = build_staging_dir(
        args.cache_dir, args.tickers_csv, args.staging_dir, args.title, dataset_id, args.license_name
    )
    n_parquet = len(list(args.cache_dir.glob("*.parquet")))
    print(f"[upload_cache_dataset] staged {n_parquet} parquet files + {args.tickers_csv.name} -> {staging_dir}")
    print(f"[upload_cache_dataset] dataset id: {dataset_id}")

    kaggle_prefix = _kaggle_cmd_prefix()
    # -t/--keep-tabular: `kaggle datasets create`/`version` convert
    # "tabular" files to CSV by default. Our parquet files were observed to
    # survive an upload unconverted anyway (checked directly against the
    # live dataset afterward), but there's no reason to rely on that --
    # this is our price cache's actual source format, not something to let
    # a CLI default silently reinterpret.
    common_flags = ["-t"]
    if args.version:
        cmd = kaggle_prefix + ["datasets", "version", "-p", str(staging_dir), "-m", args.version_notes] + common_flags
    else:
        cmd = kaggle_prefix + ["datasets", "create", "-p", str(staging_dir)] + common_flags
        # `kaggle datasets create` is private by default -- this never adds
        # -u/--public, on purpose.

    print(f"[upload_cache_dataset] command: {' '.join(cmd)}")
    if args.dry_run:
        print("[upload_cache_dataset] --dry-run: not invoking kaggle. Review the staged files above, "
              "then re-run without --dry-run (or copy/paste the command).")
        return

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # Observed directly: for a large (500+ file) dataset, the kaggle
        # CLI can print "Upload successful" for every single file, then
        # exit non-zero on a JSON-decode error in some final housekeeping
        # call, even though the dataset was actually created/updated
        # correctly (verified with `kaggle datasets files <id>` after this
        # happened -- every file was present). Don't assume this failure
        # means nothing happened; check before retrying.
        print(
            f"\n[upload_cache_dataset] kaggle exited non-zero. This can happen even when the "
            f"upload actually succeeded (seen with large datasets) -- verify before retrying:\n"
            f"  kaggle datasets files {dataset_id}\n"
        )
        raise


if __name__ == "__main__":
    main()
