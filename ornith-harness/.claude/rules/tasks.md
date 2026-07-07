---
paths: tasks/**
---
# Task corpus constraints

Applies when creating or editing task specs.

- Every task declares provenance: `public-pretrained` (R2E-Gym/SWE-Gym era), `own-repo`, or `post-cutoff` (mined after 2026-06-25).
- Holdout contains only `own-repo` and `post-cutoff` tasks. Nothing public-and-old.
- A task is admitted only after passing the determinism check: identical verdict on 3 blind reruns of its verification command.
- Verification commands are binary and self-contained: exit 0 = pass, nonzero = fail, no LLM judgment in the verdict path.
- Task repo state is a pinned container image reference, never a live checkout.
- Results are always sliceable by provenance; a change that improves `public-pretrained` but not `own-repo` is treated as contamination signal, not progress.
