# Ornith → Claude Code adaptation harness

Optimize a scaffold (Claude Code config layer) for Ornith-1.0-35B as a coding
worker, via mutate-and-select search + optional LoRA. Full spec: PLAN.md.
Goals and gates: PLAN.md §Goals. Session log and results: findings/FINDINGS.md
(read it first in every session — it is the project's memory).

## The one structural invariant

This repo's own `.claude/` is the **instrument**. Scaffold variants under
`experiments/variants/*/claude-config/` are the **subject** — data,
materialized into task containers at run time. The two never mix. Never copy
variant config into the harness config or vice versa.

## Layout

- `experiments/variants/vNNN-*/` — scaffold variants (manifest.yaml + claude-config/)
- `experiments/runs/<run_id>/` — results, transcripts, summaries (written by the Modal runner only)
- `tasks/dev/`, `tasks/holdout/` (sealed by hook), `tasks/staging/`, `tasks/schema.md`
- `infra/modal_app.py` — serving + sweep runner; `infra/budget.yaml` — spend cap
- `findings/FINDINGS.md` — append-only log; `findings/cost-ledger.csv` — runner-written

## Working agreements

- Non-trivial work runs through the `goal` skill: criteria pinned before work,
  validated after, logged to findings. Gate state: `python3 infra/status.py`.

- Kill criteria and statistical thresholds live in PLAN.md §Decision rules and
  are pre-committed; surface them, never re-litigate them mid-cycle.
- Secrets live in Modal secrets / env only. Never in files, never echoed.
- When blocked by a guard hook, the block is the correct outcome — report it.
