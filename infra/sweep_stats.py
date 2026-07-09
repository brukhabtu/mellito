#!/usr/bin/env python3
"""Aggregation for a variant eval sweep: per-task pass rates, paired comparison
vs the parent variant, provenance slices, and cost — plus the summary.json and
cost-ledger.csv writers. Pure functions over trial-result dicts, so this is
unit-testable without Modal or a GPU (run_trial produces the dicts; this turns
them into the numbers the operator approves on).

A trial result dict (one task × one trial):
  {"task": id, "trial": i, "provenance": str,
   "verdict": "pass"|"fail"|"invalid",   # invalid = execution error, per PLAN
   "tokens_in": int, "tokens_out": int,
   "gpu_seconds": float, "wall_clock_s": float}

Verdict rules (PLAN "Error ≠ fail"): invalid trials are excluded from pass-rate
denominators and from paired stats on BOTH sides of a comparison.
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

# Cost-attribution throughput constant, shared with trial_logic (the single
# source of truth). Imported so the per-trial attribution string can never drift
# from the divisor run_trial actually uses; the literal is a fallback for when
# this module is imported without trial_logic on the path.
try:
    from trial_logic import AGG_TOK_PER_S
except ImportError:
    AGG_TOK_PER_S = 908.0


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k passes of n valid trials."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def per_task(results: list) -> dict:
    """task_id -> {passes, valid, invalid, pass_rate, provenance}. A task's
    valid trials are its non-invalid trials; pass_rate is over valid only."""
    agg = defaultdict(lambda: {"passes": 0, "valid": 0, "invalid": 0,
                               "provenance": None})
    for r in results:
        t = agg[r["task"]]
        t["provenance"] = r.get("provenance")
        if r["verdict"] == "invalid":
            t["invalid"] += 1
        else:
            t["valid"] += 1
            if r["verdict"] == "pass":
                t["passes"] += 1
    for t in agg.values():
        t["pass_rate"] = (t["passes"] / t["valid"]) if t["valid"] else None
    return dict(agg)


def _solved(task_stat: dict, threshold: float = 0.5) -> bool | None:
    """A task counts as 'solved' by a variant if it passes a majority of its
    valid trials. None if the task had no valid trials (excluded from pairs)."""
    if not task_stat["valid"]:
        return None
    return task_stat["pass_rate"] >= threshold


def paired(this_pt: dict, parent_pt: dict | None) -> dict:
    """Per-task paired comparison vs parent: win/loss/tie on the solved
    indicator, over tasks where BOTH sides have >=1 valid trial. Returns counts
    and the net task delta (the metric the keep-threshold in PLAN is stated in:
    'no keeps below +5 paired')."""
    if not parent_pt:
        return {"comparable_tasks": 0, "wins": 0, "losses": 0, "ties": 0,
                "net_tasks": 0, "note": "no parent (baseline variant)"}
    wins = losses = ties = 0
    for task, ts in this_pt.items():
        ps = parent_pt.get(task)
        if ps is None:
            continue
        a, b = _solved(ts), _solved(ps)
        if a is None or b is None:
            continue
        if a and not b:
            wins += 1
        elif b and not a:
            losses += 1
        else:
            ties += 1
    return {"comparable_tasks": wins + losses + ties, "wins": wins,
            "losses": losses, "ties": ties, "net_tasks": wins - losses}


def by_provenance(per_task_stats: dict) -> dict:
    """provenance -> {tasks, solved, pass_rate_over_tasks, ci95}. The
    contamination tripwire reads these slices."""
    groups = defaultdict(list)
    for stat in per_task_stats.values():
        groups[stat["provenance"] or "unknown"].append(stat)
    out = {}
    for prov, stats in groups.items():
        valid = [s for s in stats if s["valid"]]
        solved = sum(1 for s in valid if _solved(s))
        n = len(valid)
        lo, hi = _wilson(solved, n)
        out[prov] = {"tasks": len(stats), "valid_tasks": n, "solved": solved,
                     "solved_rate": (solved / n) if n else None,
                     "ci95": [round(lo, 3), round(hi, 3)]}
    return out


def cost(results: list, usd_per_gpu_hour: float) -> dict:
    gpu_s = sum(r.get("gpu_seconds", 0.0) for r in results)
    usd = gpu_s / 3600.0 * usd_per_gpu_hour
    # api_usd is a legacy field (hosted workers removed 2026-07-09); always 0
    # on new runs, kept in the fold so historical rows still sum correctly.
    api_usd = sum(r.get("api_usd", 0.0) for r in results)
    solved_tasks = sum(1 for s in per_task(results).values() if _solved(s))
    tin = sum(r.get("tokens_in", 0) for r in results)
    tout = sum(r.get("tokens_out", 0) for r in results)
    valid = [r for r in results if r["verdict"] != "invalid"]
    return {
        "gpu_seconds": round(gpu_s, 1),
        "usd": round(usd, 4),
        "api_usd": round(api_usd, 4),
        "usd_per_solved_task": round((usd + api_usd) / solved_tasks, 4)
        if solved_tasks else None,
        # How the two spend figures are derived, for the operator reading a run.
        "gpu_attribution": {"per_trial": f"tokens_out/{AGG_TOK_PER_S:g}",
                            "ledger": "max(sum, wall)"},
        "tokens_in": tin, "tokens_out": tout,
        "trials": len(results), "valid_trials": len(valid),
        "invalid_trials": len(results) - len(valid),
        "mean_wall_s": round(sum(r.get("wall_clock_s", 0) for r in valid) / len(valid), 1)
        if valid else None,
    }


def summarize(run_id: str, variant: str, parent: str | None, results: list,
              parent_per_task: dict | None, usd_per_gpu_hour: float,
              worker_model: str | None = None) -> dict:
    pt = per_task(results)
    solved = sum(1 for s in pt.values() if _solved(s))
    valid_tasks = sum(1 for s in pt.values() if s["valid"])
    lo, hi = _wilson(solved, valid_tasks)
    out = {
        "run_id": run_id, "variant": variant, "parent": parent,
        "tasks": len(pt), "valid_tasks": valid_tasks,
        "solved_tasks": solved,
        "pass_rate_over_tasks": round(solved / valid_tasks, 4) if valid_tasks else None,
        "pass_rate_ci95": [round(lo, 3), round(hi, 3)],
        "paired_vs_parent": paired(pt, parent_per_task),
        "by_provenance": by_provenance(pt),
        "cost": cost(results, usd_per_gpu_hour),
        "per_task": {k: {"passes": v["passes"], "valid": v["valid"],
                         "invalid": v["invalid"], "pass_rate": v["pass_rate"],
                         "provenance": v["provenance"]} for k, v in pt.items()},
    }
    if worker_model is not None:
        out["worker_model"] = worker_model
    return out


def write_summary(run_dir: Path, summary: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "summary.json"
    p.write_text(json.dumps(summary, indent=2))
    return p


def append_ledger(ledger: Path, run_id: str, timestamp: str,
                  gpu_seconds: float, usd: float) -> None:
    """Append one row; write the header if the ledger is new."""
    new = not ledger.exists()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "timestamp", "gpu_seconds", "usd"])
        w.writerow([run_id, timestamp, round(gpu_seconds, 1), round(usd, 4)])
