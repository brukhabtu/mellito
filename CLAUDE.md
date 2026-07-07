# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository contents

This repo currently contains a single project, `ornith-harness/`, described
by the user as a subset extracted from a larger original harness. All
current work happens under that directory; there is no other code at the
repo root. `ornith-harness/` has its own `CLAUDE.md`, which is the primary
source of truth for working agreements in that project — read it (and
`ornith-harness/findings/FINDINGS.md`) first before making changes there.
This root file gives orientation for navigating into it; it does not
duplicate its rules.

## ornith-harness: what it is

An experiment harness to adapt `Ornith-1.0-35B` (a self-hosted open model)
into a Claude Code coding worker via scaffold mutate-and-select search
(and conditionally a LoRA pass), with the goal of beating Haiku 4.5 on
cost-per-solved-task at a comparable pass rate — or producing a documented
negative result. Full goals/gates/decision rules: `ornith-harness/PLAN.md`.
Session memory (read every session): `ornith-harness/findings/FINDINGS.md`.

Reading order for a fresh session: `PLAN.md` → `CLAUDE.md` →
`findings/FINDINGS.md`.

## Commands (run from `ornith-harness/`)

- `python3 infra/status.py` — mechanical gate checker; prints met/unmet for
  goals G1–G6 and the current frontier gate.
- `modal run infra/modal_app.py::smoke` — Phase 0 serving gate (G1).
- `modal run infra/modal_app.py::run_sweep --variant <id> --trials <n>` —
  evaluate a scaffold variant against the dev task set (`--split holdout`
  is refused unless `.holdout-unlocked` exists at repo root). Normally
  invoked via the `run-eval` skill, not directly.
- `modal deploy infra/modal_app.py` — deploy the vLLM serving endpoint.

`infra/modal_app.py` is a skeleton: `serve` is implemented; `smoke`,
`run_trial`, and `run_sweep`'s body are `TODO`-marked Phase 0/1 work.

## Architecture

**The one structural invariant:** this repo's own `.claude/` is the
*instrument* (the harness's own Claude Code config, used to do the
optimization work). `experiments/variants/*/claude-config/` is the
*subject* — data materialized into task containers at run time. The two
must never mix; never copy variant config into the harness config or vice
versa.

**Layout:**
- `experiments/variants/vNNN-*/` — scaffold variants: `manifest.yaml`
  (id, parent, hypothesis, date, status) + `claude-config/`. One mutation
  per variant, immutable once evaluated; a fix is a new child variant.
  Lineage is git, one commit per variant (`variant: vNNN <hypothesis>`).
- `experiments/runs/<run_id>/` — per-run results, transcripts, summaries;
  written only by the Modal runner, never hand-edited.
- `tasks/dev/`, `tasks/holdout/`, `tasks/staging/` — task specs, one dir
  per task (`tasks/<split>/<task-id>/task.yaml`), schema in
  `tasks/schema.md`. `holdout/` is sealed by a hook (see below) and
  contains only `own-repo`/`post-cutoff` provenance tasks; `staging/` is
  curated-but-unassigned, promoted to holdout manually by the operator.
- `infra/modal_app.py` — Modal serving + sweep runner; `infra/budget.yaml`
  — monthly GPU spend cap enforced by a hook.
- `findings/FINDINGS.md` — append-only project memory/log, one entry per
  run/mutation/admission/incident/decision, always citing a run ID.
  `findings/cost-ledger.csv` — runner-written spend log.

**Guardrails (`.claude/hooks/`, enforced automatically, not advisory):**
- `guard-holdout.py` blocks any Read/Grep/Glob/Edit/Write/Bash touching
  `tasks/holdout/**` unless the operator has created `.holdout-unlocked`
  (single gate run, logged). A block is the correct outcome — report it,
  never work around it.
- `guard-budget.py` blocks `modal run|deploy|launch` once month-to-date
  spend (from `cost-ledger.csv`) reaches the cap in `infra/budget.yaml`.
  Override is a single-shot `.budget-unlocked` file the hook deletes after
  use.

**Skills (`.claude/skills/`)**, invoked for their respective workflows
rather than reimplemented ad hoc:
- `goal` — pin acceptance criteria before non-trivial work, validate after.
- `run-eval` — materialize a variant, launch a sweep, collect results, log
  to findings, update variant status.
- `classify-failures` — classify a run's failed trajectories into a
  capability/format/harness-friction taxonomy.
- `propose-mutation` — produce exactly one new variant with a falsifiable
  hypothesis from the latest failure taxonomy.
- `corpus-curator` — admit a task into the corpus (hermeticity, 3+3
  determinism check, provenance, split legality).

**Experiment integrity constraints** (`.claude/rules/`): every reported
metric must cite a run ID; only verifier-passing trajectories are training
data candidates; comparisons between variants use paired per-task stats on
identical task sets; execution errors are `invalid`, never counted as
`fail`, and excluded from both sides of a paired comparison; surprises are
logged in `FINDINGS.md` before being worked around.
