---
paths: tasks/**
---
# Task corpus constraints

Applies when creating or editing task specs.

- Every task declares provenance: `public-pretrained` (R2E-Gym/SWE-Gym era), `own-repo`, `post-cutoff` (mined after 2026-06-25), or `held-out-public` (public repos disjoint from dev, ~2025-06 — operator-approved best-effort holdout; see tasks/schema.md).
- Holdout contains `own-repo`, `post-cutoff`, or `held-out-public` tasks — never a public task whose repo also appears in dev. `held-out-public` carries a weaker contamination guarantee than `post-cutoff`: a dev/holdout gap on it is a generalization signal, not a clean contamination verdict.
- A task is admitted only after passing the determinism check: identical verdict on 3 blind reruns of its verification command.
- Verification commands are binary and self-contained: exit 0 = pass, nonzero = fail, no LLM judgment in the verdict path.
- Task repo state is a pinned container image reference, never a live checkout.
- Results are always sliceable by provenance; a change that improves `public-pretrained` but not `own-repo` is treated as contamination signal, not progress.
