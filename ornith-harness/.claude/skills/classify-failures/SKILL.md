---
description: >-
  Classify the failed trajectories of an eval run into the three-way failure
  taxonomy (capability / format / harness-friction). Use after any eval run
  when asked to "classify failures", "analyze the failed runs", "what went
  wrong in run X", or before proposing a scaffold mutation. Reads transcripts
  in an isolated context and returns only the taxonomy table.
argument-hint: "[run-id]"
arguments: [run]
context: fork
allowed-tools: Read, Grep, Glob
---

# Classify failed trajectories for run $run

Read `experiments/runs/$run/summary.json`, then every transcript listed under
its failed trials. For each failure assign exactly one primary class:

- **capability** — the model's plan or code was wrong: misdiagnosed the bug,
  wrong fix, failed to localize. More prompting won't fix these; they are
  escalation-tier evidence.
- **format** — the intent was right but the mechanics broke: malformed tool
  call, wrong edit format, `<think>` leakage, truncation, schema mismatch.
  These are scaffold-or-LoRA fixable.
- **harness-friction** — the model fought the harness: ignored the task
  framing, tried to micro-plan instead of using its own inner loop, retried in
  degenerate ways, or stalled awaiting direction. These are task-shaping
  evidence — usually the scaffold grants too little self-direction.

Also flag (secondary tags, non-exclusive): `suspected-contamination`
(instant solve attempt with no exploration), `verifier-gaming` (touched
expected outputs rather than fixing), `timeout`.

Return ONLY:
1. A table: task_id | trial | primary class | secondary tags | one-line evidence quote reference (transcript line number, not the quote itself).
2. Class counts overall and sliced by provenance.
3. The 2–3 most common concrete failure patterns, each with task IDs.

Do not propose fixes — that is propose-mutation's job, in the main loop.
