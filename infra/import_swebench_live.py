#!/usr/bin/env python3
"""Import SWE-bench-Live instances as *best-effort* holdout tasks (staging).

SWE-bench-Live (`SWE-bench-Live/SWE-bench-Live`) continuously mines fresh
bug-fix PRs and, like SWE-bench, ships a prebuilt per-instance Docker image —
here under the `starryzhang/sweb.eval.x86_64.<id>` namespace, repo at `/testbed`
(system python, no conda). We use its 2025-06 split as holdout: the repos are
**disjoint from the dev set**, so it tests generalization to unseen repos.

IMPORTANT — provenance honesty. These tasks are 2025-06, which is NOT
post-2026-06-25, so they are NOT `post-cutoff`. Labelling them `post-cutoff`
would be a lie. They are staged with provenance **`held-out-public`**: public
repos disjoint from dev, best-effort recency. This is an OPERATOR-APPROVED
deviation from the PLAN rule "holdout = own-repo/post-cutoff" (no public dataset
has genuine post-cutoff tasks; see FINDINGS). The contamination guarantee is
weaker than a true post-cutoff holdout — treat a dev/holdout gap here as a
generalization signal, not a clean contamination verdict.

Tasks are written to tasks/staging/ only; the operator moves them into
tasks/holdout/ (the holdout guard blocks us by design).

Usage:
  python3 infra/import_swebench_live.py descriptor <instance_id>
  python3 infra/import_swebench_live.py batch <ids.json> [--target 18]
  python3 infra/import_swebench_live.py list [--month 202506]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path("/tmp/swebench")
MONTHS = ["202506", "202505", "202504"]  # newest-first; all disjoint-from-dev repos
BASE_URL = ("https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live/"
            "resolve/main/data/{month}-00000-of-00001.parquet")
# Repos already in the dev set — exclude so holdout stays disjoint from dev.
DEV_REPOS = {"django/django", "scikit-learn/scikit-learn", "astropy/astropy",
             "pylint-dev/pylint", "pytest-dev/pytest", "pydata/xarray",
             "sympy/sympy", "sphinx-doc/sphinx", "matplotlib/matplotlib",
             "psf/requests"}


def _parquet(month: str) -> Path:
    p = CACHE / f"live-{month}.parquet"
    if not p.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", BASE_URL.format(month=month), "-o", str(p)],
                       check=True)
    return p


def load_rows(months=MONTHS) -> list:
    import pyarrow.parquet as pq
    rows = []
    for m in months:
        for r in pq.read_table(str(_parquet(m))).to_pylist():
            r["_month"] = m
            rows.append(r)
    return rows


def find(rows, iid):
    for r in rows:
        if r["instance_id"] == iid:
            return r
    raise KeyError(iid)


def image_ref(iid: str) -> str:
    return "starryzhang/sweb.eval.x86_64." + iid.replace("__", "_1776_").lower()


def _f2p(row) -> list:
    v = row["FAIL_TO_PASS"]
    return v if isinstance(v, list) else json.loads(v)


def verify_cmd(row) -> str:
    """Build an offline binary verify from the instance's pytest test_cmds.
    Every SWE-bench-Live row uses log_parser=pytest, so FAIL_TO_PASS are pytest
    node ids. We keep the command's env wrapper (uv run / poetry run / python -m
    / PYTHONPATH=...) but run exactly the FAIL_TO_PASS nodes with `-rA`, so exit
    0 iff they pass — broken state fails, solution passes."""
    tc = row["test_cmds"]
    tc = tc[0] if isinstance(tc, list) and tc else str(tc)
    toks = tc.split()
    if "pytest" not in toks:
        raise ValueError(f"non-pytest test_cmd, unsupported: {tc!r}")
    prefix = " ".join(toks[:toks.index("pytest")])  # env wrapper before pytest
    nodes = " ".join(f"'{n}'" for n in _f2p(row))
    runner = (prefix + " pytest" if prefix else "pytest")
    return f"cd /testbed && {runner} -rA {nodes}"


def descriptor(row, image=None) -> dict:
    base = row["base_commit"]
    return {
        "id": row["instance_id"],
        "image": image or image_ref(row["instance_id"]),
        "workdir": "/testbed",
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
    import time
    last = None
    for attempt in range(5):
        r = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
        if r.returncode == 0:
            break
        last = r.stderr.strip()[-300:]
        time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"docker pull failed after retries: {last}")
    out = subprocess.run(
        ["docker", "inspect", "--format", "{{index .RepoDigests 0}}", image],
        capture_output=True, text=True, check=True).stdout.strip()
    return out or image


def _indent(text, n):
    pad = " " * n
    return "\n".join(pad + l for l in (text or "").splitlines()) or (pad + "(none)")


def task_yaml(row, pinned, result) -> str:
    prob = (row.get("problem_statement") or "").strip()
    return f"""id: {row['instance_id']}
provenance: held-out-public
source: "SWE-bench-Live {row['_month']} · {row['repo']} · PR#{row.get('pull_number')} · {str(row.get('created_at'))[:10]}"
image: {pinned}
description: |
{_indent(prob, 2)}
verify: {json.dumps(result['verify_display'])}
timeout_s: 1800
notes: "Best-effort holdout (operator-approved): public 2025-06, repo disjoint from dev; NOT strict post-cutoff. Weaker contamination guarantee."
admitted:
  determinism_check: {result['determinism_check']}
  date: {result['date']}
  by: import_swebench_live.py
"""


def cmd_descriptor(args):
    print(json.dumps(descriptor(find(load_rows(), args.instance_id)), indent=2))


def cmd_list(args):
    for r in load_rows([args.month] if args.month else MONTHS):
        if r["repo"] in DEV_REPOS:
            continue
        tc = r["test_cmds"]; tc = tc[0] if isinstance(tc, list) else str(tc)
        if "pytest" not in tc.split():
            continue
        print(f"{r['instance_id']}\t{r['repo']}\tf2p={len(_f2p(r))}\t{r['_month']}")


def cmd_batch(args):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    ids = json.loads(Path(args.ids).read_text())
    rows = load_rows()
    sys.path.insert(0, str(ROOT / "infra"))
    import determinism_check as dc

    lock = threading.Lock()
    admitted, rejected = [], []
    dest = ROOT / "tasks" / "staging"
    have = {p.parent.name for p in dest.glob("*/task.yaml")}
    stop = threading.Event()

    def work(iid):
        if stop.is_set() or iid in have:
            return
        image = image_ref(iid)
        try:
            row = find(rows, iid)
            pinned = pull_and_pin(image)
            desc = descriptor(row, image=image)
            result = dc.check(desc)
        except Exception as e:
            with lock:
                rejected.append((iid, f"error: {str(e)[:120]}"))
            subprocess.run(["docker", "rmi", "-f", image], capture_output=True)
            print(f"[live] {iid} REJECTED (error: {str(e)[:80]})", file=sys.stderr)
            return
        if result["admissible"]:
            result["verify_display"] = desc["verify"]
            result["date"] = args.date
            with lock:
                if len(have) < args.target:
                    out = dest / iid
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "task.yaml").write_text(task_yaml(row, pinned, result))
                    have.add(iid); admitted.append(iid)
                    n = len(have)
                    if n >= args.target:
                        stop.set()
                    print(f"[live] {iid} STAGED 6/6 ({n}/{args.target})", file=sys.stderr)
        else:
            with lock:
                rejected.append((iid, result["determinism_check"]))
            print(f"[live] {iid} REJECTED ({result['determinism_check']})", file=sys.stderr)
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, ids))
    print(json.dumps({"staged": sorted(admitted), "rejected": rejected,
                      "staging_total": len(have)}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("descriptor"); d.add_argument("instance_id")
    d.set_defaults(func=cmd_descriptor)
    l = sub.add_parser("list"); l.add_argument("--month", default=None)
    l.set_defaults(func=cmd_list)
    b = sub.add_parser("batch"); b.add_argument("ids")
    b.add_argument("--target", type=int, default=18)
    b.add_argument("--workers", type=int, default=3)
    b.add_argument("--date", default="2026-07-07")
    b.set_defaults(func=cmd_batch)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
