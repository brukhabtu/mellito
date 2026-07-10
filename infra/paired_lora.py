"""P4 re-gate: paired base-vs-LoRA delta on an identical task set.

The scaffold-search delta loop (run_sweep) compares a variant against its PARENT
VARIANT on the base model. P4's axis is different: SAME scaffold, base worker vs
LoRA worker. This computes that paired per-task delta offline from two run
summaries, using the same _solved / paired primitives run_sweep uses, so the
+>=5 keep-threshold (PLAN G4) is applied identically.

Usage:
  python3 infra/paired_lora.py <base_run_id> <lora_run_id>

Both runs must be the same variant on the same split/tasks; only worker_model
differs. Prints headline solved rates, the paired win/loss/tie/net (LoRA vs
base), and the per-task disagreements. Every number cites both run IDs.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_stats as stats

RUNS = Path(__file__).resolve().parent.parent / "experiments" / "runs"


def _load(run_id: str) -> dict:
    p = RUNS / run_id / "summary.json"
    if not p.exists():
        raise SystemExit(f"no summary: {p}")
    return json.loads(p.read_text())


def _pt(summary: dict) -> dict:
    # paired()/_solved() read {task: {"valid":.., "pass_rate":..}}.
    return {t: {"valid": v["valid"], "pass_rate": v["pass_rate"]}
            for t, v in summary["per_task"].items()}


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: paired_lora.py <base_run_id> <lora_run_id>")
    base_id, lora_id = sys.argv[1], sys.argv[2]
    base, lora = _load(base_id), _load(lora_id)

    for label, s, rid in (("base", base, base_id), ("lora", lora, lora_id)):
        if s.get("worker_model") not in ("ornith-35b", "ornith-lora"):
            print(f"WARN: {label} run {rid} worker_model={s.get('worker_model')}")
    if base["variant"] != lora["variant"]:
        print(f"WARN: variant mismatch base={base['variant']} lora={lora['variant']}")

    bpt, lpt = _pt(base), _pt(lora)
    p = stats.paired(lpt, bpt)  # this=lora, parent=base

    print(f"=== P4 paired: LoRA vs base ({base['variant']}) ===")
    print(f"base  run {base_id}: solved {base['solved_tasks']}/{base['valid_tasks']}"
          f" (rate {base['pass_rate_over_tasks']}) worker={base.get('worker_model')}")
    print(f"lora  run {lora_id}: solved {lora['solved_tasks']}/{lora['valid_tasks']}"
          f" (rate {lora['pass_rate_over_tasks']}) worker={lora.get('worker_model')}")
    print(f"paired (LoRA vs base): +{p['wins']} / -{p['losses']} / ={p['ties']}"
          f"  net {p['net_tasks']:+d}  over {p['comparable_tasks']} comparable tasks")
    print(f"KEEP GATE (PLAN G4): net >= +5 on dev  ->  "
          f"{'PASS' if p['net_tasks'] >= 5 else 'FAIL'} (net {p['net_tasks']:+d})")

    # Per-task disagreements (the wins and losses), for the FINDINGS audit trail.
    print("\n-- disagreements (task: base_solved -> lora_solved) --")
    for task in sorted(set(bpt) | set(lpt)):
        bs = stats._solved(bpt[task]) if task in bpt else None
        ls = stats._solved(lpt[task]) if task in lpt else None
        if bs is None or ls is None or bs == ls:
            continue
        tag = "WIN " if (ls and not bs) else "LOSS"
        br = bpt.get(task, {}).get("pass_rate")
        lr = lpt.get(task, {}).get("pass_rate")
        print(f"  {tag} {task}: base {br} -> lora {lr}")


if __name__ == "__main__":
    main()
