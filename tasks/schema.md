# Task spec schema

One directory per task: `tasks/<split>/<task-id>/task.yaml`.

```yaml
id: myrepo-0042            # unique, stable
provenance: own-repo        # public-pretrained | own-repo | post-cutoff
source: "myrepo issue #42" # human-readable origin
image: registry/myrepo@sha256:...   # pinned digest, never a tag
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

Splits: `dev/` (search loop), `holdout/` (sealed — hook-enforced; only
own-repo and post-cutoff provenance), `staging/` (curated but unassigned;
operator moves to holdout manually).

Contamination checklist: see corpus-curator skill — hermeticity, 3+3
determinism, honest provenance, split legality, solvability floor, no oracle
leakage.
