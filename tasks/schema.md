# Task spec schema

One directory per task: `tasks/<split>/<task-id>/task.yaml`.

```yaml
id: simstim-0042            # unique, stable
provenance: own-repo        # public-pretrained | own-repo | post-cutoff | held-out-public
source: "simstim issue #42" # human-readable origin
image: registry/simstim@sha256:...   # pinned digest, never a tag
description: |
  The accessibility tree walker returns stale nodes after a scene
  transition. Reproduce with `pytest tests/test_walker.py -k stale`,
  then fix so the full suite passes.
verify: "pytest -x -q"      # exit 0 = pass; binary; offline; no LLM in verdict path
timeout_s: 1800
notes: ""                   # curator notes; never solution hints
admitted:                   # filled by corpus-curator at admission
  determinism_check: 6/6
  date: 2026-07-06
  by: bruk
```

Splits: `dev/` (search loop), `holdout/` (sealed — hook-enforced), `staging/`
(curated but unassigned; operator moves to holdout manually).

Holdout provenance legality: `own-repo` and `post-cutoff` are the intended
categories. `held-out-public` is an **operator-approved best-effort** category
(added 2026-07-07): public repos disjoint from the dev set, mined ~2025-06 (the
freshest public benchmark available — no public dataset has genuine
post-2026-06-25 tasks; see FINDINGS). It is NOT strictly post-cutoff, so its
contamination guarantee is weaker: treat a dev/holdout gap on `held-out-public`
tasks as a **generalization** signal (unseen repos), not a clean contamination
verdict. Slice results by provenance so the two are never conflated.

Contamination checklist: see corpus-curator skill — hermeticity, 3+3
determinism, honest provenance, split legality, solvability floor, no oracle
leakage.
