# Ornith → Claude Code Adaptation: Project Plan

Adapt Ornith-1.0-35B into a high-performing Claude Code worker via the
cheapest sufficient rung: scaffold search → LoRA → (only if evidence demands)
RL. Deliberately not employer-specific; recipe and findings publishable
either way.

## North-star goal

By ~2026-09-15: route a defined class of coding tasks to self-hosted
Ornith-35B inside Claude Code at **≥30% lower cost-per-solved-task than
Haiku 4.5**, at a pass rate **within 5 points** of it on holdout — or a
documented negative result explaining why. The published recipe is a
first-class deliverable in both outcomes.

## Goals & gates

- **G1 Serving.** Ornith-35B on Modal behind an Anthropic-compatible
  endpoint. Gate: `modal run infra/modal_app.py::smoke` exits 0
  (schema-clean tool calls, no `<think>` leakage, 20/20 trivials).
- **G2 Corpus.** ≥40 dev + ≥15 holdout hermetic tasks, each passing the 3+3
  determinism check, provenance recorded. Gate: corpus manifest complete;
  holdout contains only own-repo/post-cutoff.
- **G3 Measurement.** Every trial emits verdict/tokens/gpu-seconds/wall-clock
  under a run ID; baseline table for Ornith / Haiku 4.5 / Sonnet 5 on the
  same corpus with paired stats. Gate: FINDINGS.md baseline entry exists.
- **G4 Optimized scaffold.** A variant beating v001 by ≥5 points paired on
  dev, dev/holdout gap ≤5 points, lineage in git. Gate: the single unlocked
  holdout run confirms.
- **G5 (conditional) LoRA.** Adapter converts surviving format-class
  failures without regressing smoke suite or holdout.
- **G6 Decision.** Routing recommendation with blended-cost math, kill
  criterion explicitly evaluated, write-up shipped.

## Decision rules (pre-committed — surface, don't re-litigate)

- **Minimum detectable difference:** at this corpus size (~40 tasks × 5
  trials) treat paired improvements <5 points as noise. No variant is kept
  below that; a "trend" is not a keep.
- **Stopping rule (Phase 3):** stop the search after 3 consecutive
  non-keeping mutations, then run the holdout gate.
- **Kill criterion:** if after one scaffold+LoRA alternation cycle,
  cost-per-solved-task doesn't beat Haiku 4.5 by ≥30% OR dev pass rate is
  >10 points behind Haiku, the project ships as a negative result +
  write-up. No extension without new outside evidence.
- **Contamination tripwire:** improvement on public-pretrained tasks that
  isn't mirrored on own-repo tasks is treated as contamination, not
  progress.
- **Error ≠ fail:** execution errors are `invalid`, excluded from stats for
  both sides of a pair, and logged.

## User stories

**Operator (Bruk)**
- Start a sweep with one command and walk away; hooks guarantee no holdout
  reads and no budget overruns.
- See per variant: paired result with CI, $/solved-task, wall-clock,
  lineage — approvable in <5 minutes.
- Kill criteria pre-committed once; the system reports "met / not met" each
  cycle.
- Resume in a fresh session from FINDINGS.md alone.

**Executor (Claude Code)**
- Materialize any variant into task workspaces without touching harness
  config (structural: variants are data under experiments/).
- Classify failure batches in a fork, returning only the taxonomy.
- Propose exactly one mutation with a written hypothesis; structurally
  unable to cite numbers without run IDs or to peek at holdout.
- Record blockages and surprises as data, never improvise around them.

**Future adopter**
- Point the harness at their own repo by writing task specs against
  tasks/schema.md; nothing else changes.
- Reconstruct every keep/reject decision from FINDINGS.md.
- Swap the worker model behind one config value and rerun the corpus.

## Corpus strategy

Ornith's own training data is unpublished — assume it saw everything public
and old. Blend:
- **Dev (~40):** ~half R2E-Gym / SWE-Gym imports (cheap volume; memorization
  inflates all variants roughly equally under paired comparison), ~half
  own-repo (simstim first — 199 tests; Kerf; keyboard tooling), optionally
  generated at volume via SWE-smith against own repos.
- **Holdout (≥15):** SWE-rebench tasks mined post-2026-06-25 + reserved
  own-repo bugs. Sealed by hook; single unlocked run at G4.
- Every task through corpus-curator's admission checklist; provenance on
  every result slice.

## Phases

- **P0 Serving (weekend).** Modal app: vLLM ≥0.19.1, FP8 weights volume,
  patched chat template (shipped jinja has breaking asserts), qwen3_xml
  tool parser + qwen3 reasoning parser, prefix caching, short scaledown.
  Anthropic-compat proxy. Gate G1.
- **P1 Corpus (days, not the week originally planned).** Import + curate per
  strategy above. Gate G2.
- **P2 Baselines (a day of runs).** Ornith v001, Haiku 4.5, Sonnet 5 on dev;
  full trajectory logging from run one (it is future training data). Gate G3.
- **P3 Scaffold search (2–3 weeks; most expected gains).** Loop:
  run-eval → classify-failures → propose-mutation → operator keeps/rejects.
  Highest-leverage axis: degree of worker self-direction (Ornith's weights
  expect to write their own inner loop). Operator remains the selection step
  for at least the first two cycles; promote to autonomous only after the
  guardrails have caught a real mistake.
- **P4 LoRA (conditional on format-class residue).** Unsloth QLoRA on
  passing trajectories under the winning scaffold; `<think>` preserved in
  targets; rank 32 / α 64; attention + shared layers only, router excluded.
  Re-gate smoke + dev + holdout.
- **P5 Alternate.** Reopen P3 briefly on new weights; 1–2 alternations is
  realistic convergence. Then G6.

## Deferred: RL rung design notes

Token-level GRPO (not "AGPO" — misattribution), DAPO-style asymmetric clip
(start ε⁻=0.2, ε⁺=0.28); async pipeline-RL staleness weight w(d_t) (start
K1≈1, K2≈4–8 update steps — Ornith's values undisclosed). Three-layer
anti-hacking: immutable boundary / deterministic monitor (zero reward AND
excluded from GRPO group; if exclusion collapses groups, switch to shaped
penalties) / frozen LLM judge veto. Train in an open Claude Code-shaped
harness, evaluate in real Claude Code. The P1–P4 byproducts (environments,
trajectories, eval harness) are ~80% of RL infra.

## Standing cautions

- All Ornith benchmarks are vendor-reported and unreproduced; no ablation
  separates self-scaffolding from base-model quality. Trust only our runs.
- Sonnet 5 ($2/$10 intro to Aug 31, then $3/$15) is the escalation tier and
  raises the bar the local worker must clear.
- Secrets in Modal secrets/env only. Repo assumes it may go public.
