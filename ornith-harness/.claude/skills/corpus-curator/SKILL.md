---
description: >-
  Convert a bug, issue, or imported benchmark task into a hermetic task spec
  for the corpus. Use when asked to "add a task", "import tasks from R2E-Gym
  or SWE-Gym or SWE-rebench", "turn this own-repo bug into a task", or when
  building out the dev/holdout sets. Owns the admission checklist.
argument-hint: "[source] [dev|holdout]"
arguments: [source, split]
---

# Curate a task into the corpus

Source: `$source` → split: `$split`

Schema (tasks/schema.md) in brief: `task.yaml` with id, provenance
(public-pretrained | own-repo | post-cutoff), image (pinned digest, never a
tag), description (what to fix/build, no solution hints), verify (single
command, exit code is the verdict), timeout_s, notes.

## Admission checklist — every item, every task

1. **Hermeticity**: repo state is a container image pinned by digest;
   `verify` runs offline inside it.
2. **Determinism**: run `verify` 3× against the known-good solution and 3×
   against the broken state — verdicts must be 6/6 consistent. Flaky → reject
   or fix the test, never admit.
3. **Provenance**: assign honestly. If the repo existed publicly before
   2026-06-25 and the fix is in its history, it is `public-pretrained`
   regardless of how it was imported.
4. **Split legality**: holdout admits only `own-repo` and `post-cutoff`.
   Never write into tasks/holdout/ directly — stage the spec in
   tasks/staging/ and let the operator move it (the holdout guard blocks you
   anyway; that block is correct).
5. **Solvability floor**: the task description must contain enough signal
   that a strong model *could* solve it — if it requires tribal knowledge not
   present in the repo, enrich the description or reject.
6. **No oracle leakage**: the description and workspace must not contain the
   expected diff, the fixing commit message, or test names that give away the
   answer.

For batch imports from R2E-Gym / SWE-Gym / SWE-rebench: sample-check items
3, 5, 6 on at least 20% of the batch; run item 2 on every task admitted.
Log every admission and rejection (with reason) to findings/FINDINGS.md.
