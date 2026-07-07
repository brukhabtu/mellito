#!/usr/bin/env python3
"""Mechanical gate checker for PLAN.md goals G1-G6.

Stateless: reads the repo, prints met/unmet per gate and the frontier (the
next gate whose prerequisites are met). Judgment-shaped gates report the
mechanical part and mark the rest MANUAL.

Usage: python3 infra/status.py
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check(cond, label, detail=""):
    mark = "PASS" if cond else "----"
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))
    return bool(cond)


def count_tasks(split):
    return len(list((ROOT / "tasks" / split).glob("*/task.yaml")))


def findings_has(pattern):
    """Search only real log entries: content after the '---' separator that
    closes the schema header in FINDINGS.md."""
    f = ROOT / "findings" / "FINDINGS.md"
    if not f.exists():
        return False
    text = f.read_text()
    _, sep, entries = text.partition("\n---\n")
    if not sep:
        entries = text  # separator removed; fail open to whole file
    return re.search(pattern, entries, re.M) is not None


def main() -> int:
    gates = {}

    print("G1 Serving")
    smoke = subprocess.run(
        ["modal", "run", "infra/modal_app.py::smoke"],
        cwd=ROOT, capture_output=True, text=True,
    ) if shutil_which("modal") else None
    gates["G1"] = check(
        smoke is not None and smoke.returncode == 0,
        "smoke suite exits 0",
        "modal CLI missing" if smoke is None else f"exit {smoke.returncode}",
    )

    print("G2 Corpus")
    dev, hold = count_tasks("dev"), count_tasks("holdout")
    g2a = check(dev >= 40, f"dev tasks >= 40", f"{dev}")
    g2b = check(hold >= 15, f"holdout tasks >= 15", f"{hold} (count only; contents sealed)")
    gates["G2"] = g2a and g2b

    print("G3 Measurement")
    ledger = ROOT / "findings" / "cost-ledger.csv"
    g3a = check(ledger.exists() and sum(1 for _ in csv.reader(ledger.open())) > 1,
                "cost ledger has entries")
    g3b = check(findings_has(r"baseline"), "FINDINGS.md baseline entry")
    gates["G3"] = g3a and g3b

    print("G4 Optimized scaffold")
    g4a = check(findings_has(r"verdict:\s*kept"), "at least one kept variant")
    print("  [MANUAL] paired +>=5 on dev; dev/holdout gap <=5 (verify in FINDINGS.md)")
    gates["G4"] = g4a

    print("G5 LoRA (conditional)")
    print("  [MANUAL] only if format-class residue survived P3")
    gates["G5"] = True  # conditional; never blocks the frontier

    print("G6 Decision")
    gates["G6"] = check(findings_has(r"entry-type.*decision|## .*decision"),
                        "decision entry in FINDINGS.md")

    order = ["G1", "G2", "G3", "G4", "G6"]
    frontier = next((g for g in order if not gates[g]), None)
    print(f"\nFrontier: {frontier or 'all gates met — write it up'}")
    return 0


def shutil_which(cmd):
    import shutil
    return shutil.which(cmd)


if __name__ == "__main__":
    sys.exit(main())
