# Experiment integrity

Applies to all work in this repo.

- Every reported metric cites a run ID from experiments/runs/. Numbers without a run ID do not exist.
- Failed trajectories never enter training data as imitation targets; they may
  serve as explicitly-negative examples in preference-based training (P7+,
  operator-approved 2026-07-10). Verifier-passing AND test-edit-clean
  trajectories (worker.diff touches no test file) are the only POSITIVE
  candidates. Invalid trials enter nothing.
- The experiment harness config (this .claude/, root CLAUDE.md) is never edited during an active search cycle. Harness changes happen between cycles, in their own commit, tagged `harness:`.
- tasks/holdout/** is sealed (hook-enforced). A blocked access attempt is recorded as a data point, never worked around.
- Comparisons between variants use paired per-task statistics on identical task sets, never raw pass rates across different sets.
- Anything surprising (task flake, tool failure, unexpected model behaviour) is logged in findings/FINDINGS.md before it is worked around.
