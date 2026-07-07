---
description: >-
  Propose the next scaffold mutation from the latest failure taxonomy and
  variant lineage. Use when asked to "propose a mutation", "what should we try
  next", "create the next variant", or after classify-failures completes.
  Produces exactly one new variant directory with a falsifiable hypothesis.
---

# Propose one scaffold mutation

Variant lineage and outcomes:
!`grep -h -A2 "^id:" experiments/variants/*/manifest.yaml 2>/dev/null | head -60 || echo "no variants yet"`

## Procedure

1. Read the newest failure taxonomy (findings/FINDINGS.md) and the current
   best variant's claude-config/.
2. Choose the single highest-leverage change. Priority order: harness-friction
   patterns (task framing / self-direction axis) > format patterns (tool
   exemplars, output constraints) > anything else. Never target capability
   failures with prompt changes — mark those tasks escalation-tier instead.
3. Write the hypothesis BEFORE the edit, as a falsifiable sentence naming the
   failure pattern, the change, and the expected effect:
   "vNNN: adding a worked str_replace exemplar will convert ≥half of the
   'malformed edit' format failures (tasks 12, 19, 31) to passes."
4. Create `experiments/variants/vNNN-<slug>/`: copy the parent's
   claude-config/, apply the ONE change, write manifest.yaml (id, parent,
   hypothesis, date, status: proposed).
5. Stop. Do not run the eval — the operator (or an explicit instruction)
   triggers run-eval. Proposing and scoring in one breath is how loops
   overfit.

Constraints (also enforced by the variants rule): one conceptual change; a
mutation to prose AND tools AND framing is three variants, not one. If the
taxonomy supports several equally strong candidates, present them and ask —
do not create multiple variants speculatively.
