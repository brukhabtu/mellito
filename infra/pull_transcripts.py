#!/usr/bin/env python3
"""Pull a run's per-trial artifacts (transcript.jsonl, worker.diff,
verdict.json, worker.stderr.log) from the `ornith-runs` Modal volume down to
`experiments/runs/<run_id>/` so the read-only analysis tools — classify-failures
and the trajectory-analyst — can read them locally.

run_sweep writes summary.json + trials.jsonl locally, but run_trial writes the
bulky per-trial artifacts to the volume (they don't belong in git). Analysis
needs them local; this is the bridge. Used both interactively and by the CI
classify workflow (where Modal auth comes from repo secrets).

The whole run is pulled in one call — transcripts are ~30 KB each (~10-20 MB for
a 40×5 run), so there's no point being selective. `modal volume get` only
recurses a directory when the remote has a trailing slash and the local dest is
an existing directory; both are handled here.

Usage:
  python3 infra/pull_transcripts.py <run_id> [--dest experiments/runs]
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOLUME = "ornith-runs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_id")
    ap.add_argument("--dest", default=str(ROOT / "experiments" / "runs"),
                    help="parent dir; the run lands at <dest>/<run_id>/")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    # trailing slash on the remote + existing local dir => recursive pull into
    # <dest>/<run_id>/<task>/trial<N>/...
    r = subprocess.run(
        ["modal", "volume", "get", "--force", VOLUME, f"{args.run_id}/", str(dest)],
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"pull failed: {r.stderr.strip()[-300:]}")

    run_dir = dest / args.run_id
    n = len(list(run_dir.glob("*/trial*/transcript.jsonl")))
    print(f"pulled {n} transcripts into {run_dir}")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
