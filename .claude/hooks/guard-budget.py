#!/usr/bin/env python3
"""PreToolUse guard on Bash: gate `modal run|deploy` behind a monthly GPU cap.

Reads the month-to-date spend from findings/cost-ledger.csv (written by the
Modal runner after every run: run_id,timestamp,gpu_seconds,usd) and blocks new
Modal invocations once the cap in infra/budget.yaml is reached. Exit 2 blocks.

Override: human operator creates .budget-unlocked (logged, single-shot: the
hook deletes it after one use).
"""
import csv
import json
import os
import sys
from datetime import datetime, timezone

DEFAULT_CAP_USD = 150.0  # conservative default if budget.yaml is missing


def month_spend(ledger_path: str) -> float:
    if not os.path.exists(ledger_path):
        return 0.0
    now = datetime.now(timezone.utc)
    total = 0.0
    with open(ledger_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ts = datetime.fromisoformat(row["timestamp"])
                if ts.year == now.year and ts.month == now.month:
                    total += float(row["usd"])
            except (KeyError, ValueError):
                continue
    return total


def read_cap(budget_path: str) -> float:
    if not os.path.exists(budget_path):
        return DEFAULT_CAP_USD
    with open(budget_path) as f:
        for line in f:
            if line.strip().startswith("monthly_cap_usd:"):
                try:
                    return float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
    return DEFAULT_CAP_USD


def main() -> int:
    event = json.load(sys.stdin)
    cmd = (event.get("tool_input") or {}).get("command", "")
    if not any(s in cmd for s in ("modal run", "modal deploy", "modal launch")):
        return 0

    root = os.environ.get("CLAUDE_PROJECT_DIR", ".")
    unlock = os.path.join(root, ".budget-unlocked")
    if os.path.exists(unlock):
        os.remove(unlock)  # single-shot
        return 0

    cap = read_cap(os.path.join(root, "infra", "budget.yaml"))
    spent = month_spend(os.path.join(root, "findings", "cost-ledger.csv"))
    if spent < cap:
        return 0

    print(
        f"BLOCKED: month-to-date GPU spend ${spent:.2f} >= cap ${cap:.2f} "
        f"(infra/budget.yaml). Stop and report to the operator; do not retry "
        f"or route around this. The operator can raise the cap or create "
        f".budget-unlocked for a single approved run.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
