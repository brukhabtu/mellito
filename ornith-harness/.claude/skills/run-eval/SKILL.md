---
description: >-
  Run an evaluation sweep for a scaffold variant against the dev task set on
  Modal. Use whenever asked to "run the eval", "evaluate variant vNNN",
  "get numbers for", "re-run dev set", or after a new variant is created and
  needs scoring. Handles materialization, fan-out, collection, and findings
  logging end to end.
argument-hint: "[variant-id] [trials-per-task]"
arguments: [variant, trials]
allowed-tools: Bash(modal *), Bash(python3 infra/*), Read, Write
---

# Run an evaluation sweep

Variant under test: `$variant` (default: latest in experiments/variants/).
Trials per task: `$trials` (default: 5 — do not reduce below 3; paired stats need it).

Current variant inventory:
!`ls experiments/variants/ 2>/dev/null || echo "no variants yet"`

Month-to-date spend (the budget hook will block if over cap):
!`python3 -c "import csv,datetime;n=datetime.datetime.now();print(round(sum(float(r['usd']) for r in csv.DictReader(open('findings/cost-ledger.csv')) if r['timestamp'][:7]==n.strftime('%Y-%m')),2))" 2>/dev/null || echo "0.00 (no ledger yet)"` USD

## Procedure

1. Read `experiments/variants/$variant/manifest.yaml`. If status is not
   `proposed` or `evaluated`, stop and report — evaluated variants are immutable.
2. Estimate cost: tasks × trials × mean tokens-per-trajectory (from the last
   run's summary if one exists). Report the estimate before launching.
3. Launch: `modal run infra/modal_app.py::run_sweep --variant $variant --trials $trials`.
   The runner materializes `claude-config/` as `.claude/` inside each task
   container, executes trials in parallel, and writes
   `experiments/runs/<run_id>/` (per-trial: verdict, tokens, gpu_seconds,
   wall_clock, transcript path) plus appends findings/cost-ledger.csv.
   Do not re-implement any of that logic here — it is code, not procedure.
4. When the run completes, read `experiments/runs/<run_id>/summary.json` and
   report: paired per-task comparison vs the variant's parent (win/loss/tie
   per task), pass rate ±95% CI, cost per solved task, wall-clock per task,
   all sliced by provenance class.
5. Append one row to findings/FINDINGS.md per the schema at the top of that
   file, citing the run ID. Update the variant's manifest status to `evaluated`.

Failure handling: if any task errors rather than fails (container crash, tool
error), mark it `invalid` in the summary, exclude it from stats for BOTH
variants in the pair, and log it in FINDINGS.md. Never count an error as a fail.
