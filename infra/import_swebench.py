#!/usr/bin/env python3
"""Import SWE-bench (Verified) instances into the corpus as hermetic tasks.

Each SWE-bench instance ships as a prebuilt per-instance Docker image on Docker
Hub (`swebench/sweb.eval.x86_64.<id>`), with the repo checked out at
`base_commit` under `/testbed` and a conda env `testbed` holding the deps. That
makes every instance a ready-made hermetic task — no image build needed. This
module turns one instance into the JSON descriptor that
`infra/determinism_check.py` consumes, and (in `admit` mode) drives the full
3+3 admission: pull image -> pin digest -> determinism check -> write task.yaml.

Provenance is honest and fixed: every SWE-bench instance predates the
2026-06-25 cutoff and is public, so provenance is **public-pretrained**. Such
tasks are dev-only; split legality (corpus-curator item 4) forbids them in
holdout. This importer refuses `--split holdout`.

States, mapped onto the determinism descriptor:
  base      = image's /testbed at base_commit (already there; we hard-reset)
  to_broken = apply the instance's `test_patch`  (adds the failing tests)
  to_solution = apply the instance's gold `patch` (the fix)
  verify    = run the FAIL_TO_PASS node ids under the testbed env
So: broken state fails FAIL_TO_PASS 3/3, solution passes 3/3 -> 6/6 admissible.

Usage:
  python3 infra/import_swebench.py descriptor <instance_id> [--parquet P]
  python3 infra/import_swebench.py admit <instance_id> [--split dev] [--prune]
  python3 infra/import_swebench.py list [--repo R] [--max-f2p N] [--max-p2p N]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = Path("/tmp/swebench/verified.parquet")
PARQUET_URL = (
    "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/"
    "resolve/main/data/test-00000-of-00001.parquet"
)
CUTOFF = "2026-06-25"  # PLAN.md: anything public before this is public-pretrained


def _ensure_parquet(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", PARQUET_URL, "-o", str(path)], check=True)
    return path


def load_rows(parquet: Path) -> list:
    import pyarrow.parquet as pq
    return pq.read_table(str(_ensure_parquet(parquet))).to_pylist()


def find(rows: list, instance_id: str) -> dict:
    for r in rows:
        if r["instance_id"] == instance_id:
            return r
    raise KeyError(f"{instance_id} not in dataset")


def image_ref(instance_id: str) -> str:
    """Docker Hub prebuilt-image name for a SWE-bench instance."""
    return "swebench/sweb.eval.x86_64." + instance_id.replace("__", "_1776_").lower()


def verify_cmd(row: dict) -> str:
    """Build the offline verify command using SWE-bench's own per-repo test
    runner (django uses runtests.py, most use pytest, etc.) applied to this
    instance's test directives. Exit 0 iff those tests pass. The directives
    scope covers exactly FAIL_TO_PASS ∪ PASS_TO_PASS, so: broken state -> a
    FAIL_TO_PASS test fails -> exit!=0; solution state -> all pass -> exit 0.
    Run under the image's `testbed` conda env."""
    from swebench.harness.test_spec.python import (
        MAP_REPO_VERSION_TO_SPECS, get_test_directives)
    spec = MAP_REPO_VERSION_TO_SPECS[row["repo"]][row["version"]]
    tc = spec["test_cmd"]
    test_cmd = tc if isinstance(tc, str) else " && ".join(tc)
    directives = " ".join(get_test_directives(row))
    return (
        "source /opt/miniconda3/bin/activate testbed && "
        f"cd /testbed && {test_cmd} {directives}"
    )


def tox_based(row: dict) -> bool:
    from swebench.harness.test_spec.python import MAP_REPO_VERSION_TO_SPECS
    tc = MAP_REPO_VERSION_TO_SPECS[row["repo"]][row["version"]]["test_cmd"]
    return "tox" in (tc if isinstance(tc, str) else " ".join(tc))


def descriptor(row: dict, image: str | None = None) -> dict:
    base = row["base_commit"]
    return {
        "id": row["instance_id"],
        "image": image or image_ref(row["instance_id"]),
        "workdir": "/testbed",
        # Reach a clean base state. The image is already at base_commit, but a
        # hard reset makes the starting point explicit and rerun-proof.
        "setup": [f"git reset --hard {base}", "git clean -fd"],
        "to_broken": ["git apply -v /tmp/swe_test.patch"],
        "to_solution": ["git apply -v /tmp/swe_gold.patch"],
        "files": {
            "/tmp/swe_test.patch": row["test_patch"],
            "/tmp/swe_gold.patch": row["patch"],
        },
        "verify": verify_cmd(row),
        "timeout_s": 1800,
    }


def pull_and_pin(image: str) -> str:
    """Pull the image and return its `repo@sha256:...` pinned reference.
    Docker Hub's anonymous endpoint 503s / rate-limits intermittently, so
    retry with exponential backoff before giving up."""
    import time
    last = None
    for attempt in range(5):
        r = subprocess.run(["docker", "pull", image],
                           capture_output=True, text=True)
        if r.returncode == 0:
            break
        last = r.stderr.strip()[-300:]
        time.sleep(2 ** attempt)  # 1,2,4,8,16s
    else:
        raise RuntimeError(f"docker pull failed after retries: {last}")
    out = subprocess.run(
        ["docker", "inspect", "--format", "{{index .RepoDigests 0}}", image],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return out or image


def task_yaml(row: dict, pinned_image: str, result: dict) -> str:
    prob = (row["problem_statement"] or "").strip()
    # Keep the description as the upstream problem statement (no solution
    # hints; SWE-bench statements are the issue text, not the diff).
    return f"""id: {row['instance_id']}
provenance: public-pretrained
source: "SWE-bench Verified · {row['repo']} · base {row['base_commit'][:12]}"
image: {pinned_image}
description: |
{_indent(prob, 2)}
verify: {json.dumps(result['verify_display'])}
timeout_s: 1800
notes: "Imported from SWE-bench_Verified; FAIL_TO_PASS is the verdict set."
admitted:
  determinism_check: {result['determinism_check']}
  date: {result['date']}
  by: import_swebench.py
"""


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.splitlines()) or (pad + "(none)")


def cmd_descriptor(args):
    rows = load_rows(args.parquet)
    print(json.dumps(descriptor(find(rows, args.instance_id)), indent=2))


def cmd_list(args):
    rows = load_rows(args.parquet)
    for r in rows:
        f2p, p2p = json.loads(r["FAIL_TO_PASS"]), json.loads(r["PASS_TO_PASS"])
        if args.repo and r["repo"] != args.repo:
            continue
        if len(f2p) > args.max_f2p or len(p2p) > args.max_p2p:
            continue
        print(f"{r['instance_id']}\t{r['repo']}\tf2p={len(f2p)}\tp2p={len(p2p)}")


def cmd_admit(args):
    if args.split == "holdout":
        sys.exit("refused: SWE-bench is public-pretrained; holdout is own-repo/"
                 "post-cutoff only (corpus-curator item 4). Use --split dev.")
    rows = load_rows(args.parquet)
    row = find(rows, args.instance_id)
    image = image_ref(row["instance_id"])
    print(f"[admit] {row['instance_id']} — pulling {image}", file=sys.stderr)
    pinned = pull_and_pin(image)

    desc = descriptor(row, image=image)  # check against the local tag
    sys.path.insert(0, str(ROOT / "infra"))
    import determinism_check as dc
    result = dc.check(desc)
    print(json.dumps(result, indent=2))

    if not result["admissible"]:
        if args.prune:
            subprocess.run(["docker", "rmi", "-f", image], capture_output=True)
        sys.exit(f"[admit] {row['instance_id']} NOT admissible "
                 f"({result['determinism_check']}); not written.")

    result["verify_display"] = desc["verify"]
    result["date"] = args.date
    out_dir = ROOT / "tasks" / args.split / row["instance_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task.yaml").write_text(task_yaml(row, pinned, result))
    print(f"[admit] wrote {out_dir/'task.yaml'} — {result['determinism_check']}",
          file=sys.stderr)
    if args.prune:
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True)


def cmd_batch(args):
    if args.split == "holdout":
        sys.exit("refused: SWE-bench is public-pretrained; holdout is dev-only.")
    import threading
    from concurrent.futures import ThreadPoolExecutor
    ids = json.loads(Path(args.ids).read_text())
    rows = load_rows(args.parquet)
    sys.path.insert(0, str(ROOT / "infra"))
    import determinism_check as dc

    lock = threading.Lock()
    admitted, rejected = [], []
    have = {p.parent.name for p in (ROOT / "tasks" / args.split).glob("*/task.yaml")}
    stop = threading.Event()

    def work(iid):
        if stop.is_set() or iid in have:
            return
        image = image_ref(iid)
        try:
            row = find(rows, iid)
            pinned = pull_and_pin(image)
            desc = descriptor(row, image=image)
            result = dc.check(desc)  # unique container/snapshot names -> parallel-safe
        except Exception as e:  # pull/patch/setup failure -> not admissible, logged
            with lock:
                rejected.append((iid, f"error: {str(e)[:120]}"))
            subprocess.run(["docker", "rmi", "-f", image], capture_output=True)
            print(f"[batch] {iid} REJECTED (error: {str(e)[:80]})", file=sys.stderr)
            return
        if result["admissible"]:
            result["verify_display"] = desc["verify"]
            result["date"] = args.date
            with lock:
                if len(have) < args.target:
                    out = ROOT / "tasks" / args.split / iid
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "task.yaml").write_text(task_yaml(row, pinned, result))
                    have.add(iid)
                    admitted.append(iid)
                    n = len(have)
                    if n >= args.target:
                        stop.set()
                    print(f"[batch] {iid} ADMITTED 6/6 ({n}/{args.target})",
                          file=sys.stderr)
        else:
            with lock:
                rejected.append((iid, result["determinism_check"]))
            print(f"[batch] {iid} REJECTED ({result['determinism_check']})",
                  file=sys.stderr)
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, ids))

    print(json.dumps({"admitted": sorted(admitted), "rejected": rejected,
                      "dev_total": len(have)}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("descriptor"); d.add_argument("instance_id")
    d.set_defaults(func=cmd_descriptor)

    l = sub.add_parser("list")
    l.add_argument("--repo"); l.add_argument("--max-f2p", type=int, default=3)
    l.add_argument("--max-p2p", type=int, default=10)
    l.set_defaults(func=cmd_list)

    a = sub.add_parser("admit")
    a.add_argument("instance_id")
    a.add_argument("--split", default="dev", choices=["dev", "staging"])
    a.add_argument("--prune", action="store_true",
                   help="docker rmi the image after admission (disk-bound runs)")
    a.add_argument("--date", default="2026-07-07")
    a.set_defaults(func=cmd_admit)

    b = sub.add_parser("batch")
    b.add_argument("ids", help="JSON file: list of instance_ids to try")
    b.add_argument("--split", default="dev", choices=["dev", "staging"])
    b.add_argument("--target", type=int, default=40, help="stop at this many admitted")
    b.add_argument("--workers", type=int, default=3, help="concurrent admissions")
    b.add_argument("--date", default="2026-07-07")
    b.set_defaults(func=cmd_batch)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
