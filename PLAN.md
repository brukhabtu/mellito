# Ornith → Claude Code Adaptation: Project Plan

Adapt Ornith-1.0-35B into a high-performing Claude Code worker via the
cheapest sufficient rung: scaffold search → LoRA → (only if evidence demands)
RL. Deliberately general-purpose; recipe and findings publishable
either way.

## North-star goal

By ~2026-09-15: route a defined class of coding tasks to self-hosted
Ornith-35B inside Claude Code where the improved system (scaffold and/or
weights) **beats stock Ornith + Claude Code by ≥+5 paired tasks on dev,
confirmed on holdout** — or a documented negative result explaining why.
Cost-per-solved-task on our own GPU stays a tracked observability metric, not
a comparison bar (hosted reference baselines removed by operator redirect,
2026-07-09). The published recipe is a first-class deliverable in both
outcomes.

## Goals & gates

- **G1 Serving.** Ornith-35B on Modal behind an Anthropic-compatible
  endpoint. Gate: `modal run infra/modal_app.py::smoke` exits 0
  (schema-clean tool calls, no `<think>` leakage, 20/20 trivials).
- **G2 Corpus.** ≥40 dev + ≥15 holdout hermetic tasks, each passing the 3+3
  determinism check, provenance recorded. Gate: corpus manifest complete;
  holdout contains only own-repo/post-cutoff.
- **G3 Measurement.** Every trial emits verdict/tokens/gpu-seconds/wall-clock
  under a run ID; baseline table for Ornith on the corpus with paired stats.
  Gate: FINDINGS.md baseline entry exists.
- **G4 Optimized scaffold.** A variant beating v001 by ≥5 points paired on
  dev, dev/holdout gap ≤5 points, lineage in git. Gate: the single unlocked
  holdout run confirms.
- **G5 (conditional) LoRA.** Adapter converts surviving format-class
  failures without regressing smoke suite or holdout.
- **G6 Decision.** Base-vs-tuned verdict with $/solved-task observability
  math, kill criterion explicitly evaluated, write-up shipped.

## Status (2026-07-08)

Live gate state is mechanical: `python3 infra/status.py`. Full log:
findings/FINDINGS.md. Summary of where the project stands:

- **G1 Serving — MET.** Ornith-1.0-35B-FP8 on 1×H100 (vLLM 0.24, Modal). Smoke
  20/20, 0 `<think>` leaks, schema-clean tool call (run `ap-RW4x5gYvUMJZb9c0ZwrnmF`).
  P2 throughput tuned in-place: `@modal.concurrent` + cudagraph/compile lifted
  aggregate throughput **13 → ~908 tok/s** (~70×), making a full sweep ~$5–15.
- **G2 Corpus — dev MET, holdout staged.** 40 dev tasks imported from SWE-bench
  Verified (prebuilt images, 6/6 determinism, hidden test_patch persisted so
  they run as real evals), rebalanced across 6 repos. Holdout: 18 best-effort
  tasks staged from SWE-bench-Live 2025-06 (provenance `held-out-public`).
  Deviations from the original strategy, both operator-approved and logged:
  (a) dev is 100% public-pretrained — own repos are off-limits
  for this project; (b) no public dataset has genuine post-2026-06-25 tasks, so
  holdout is a best-effort *held-out-repos* proxy, not a strict post-cutoff set
  (weaker contamination guarantee — see tasks/schema.md). Gate closes when the
  operator moves ≥15 staged specs into tasks/holdout/.
- **G3 Measurement — runner validated, Ornith baseline running.** `run_trial`
  (modal.Sandbox per task + Claude Code CLI vs the endpoint via a LiteLLM
  Anthropic-compat proxy), `run_sweep`, and `sweep_stats.py` are built,
  adversarially reviewed (10 confirmed bugs fixed), and validated end-to-end: a
  proof-of-one PASS (Ornith solved django-10973 with a real source fix, 27
  turns) and a clean 3×3 mini-sweep (0 invalid across django/pylint/astropy).
  The full 40×5 **Ornith v001 baseline** is DONE (run `20260707T215242-v001-baseline`):
  **20/40 dev tasks solved = 50%** (CI [35, 65]), $0.086/solved, 1 invalid/200.
  This is the reference every scaffold variant pairs against.
- **G4 Optimized scaffold — P3 search converged, gate NOT met by scaffold
  alone.** Three mutations tried, all rejected on the pre-committed +5 MDD floor
  (paired vs v001): v002-completion-contract (self-direction prose) **+3**,
  v003-stop-hook-enforce (mechanical empty-diff gate) **+2**, v004-localization-
  discipline (trace-before-edit prose) **0**. The stopping rule (3 consecutive
  non-keeps) has fired. Key result: **scaffold complexity is inversely correlated
  with Ornith's performance** — the shortest high-agency prompt (v002) is the
  peak at +3, every added mechanism/instruction did worse, and the targeted
  wrong-file capability cluster proved completely prompt-immune (v004 moved it
  zero). The residual failure mass is model-level (43% capability + variant-
  independent greeting-resets and a `result:""` SDK-stall floor), not scaffold-
  addressable. No variant cleared the keep bar, so there is nothing to confirm on
  the single holdout run; G4 stays open. Runs: `20260708T004754` (v002 clean),
  `20260708T015417` (v003), `20260708T030102` (v004). See findings/FINDINGS.md.
- **G5 LoRA — RAN, gate NOT met (LoRA not kept).** A rank-32 **bf16** LoRA
  (not QLoRA — Unsloth advises against 4-bit for this Qwen3.5-MoE hybrid;
  attention-only q/k/v/o_proj, router + GDN/Mamba mixer excluded) was trained on
  the 89-example v002 think-preserving SFT set (adapter `lora-20260708T162249`,
  train loss 0.258) and served on the FP8 base via vLLM `--lora-modules` (no
  merge/requantize needed). Smoke re-gate clean (20/20, 0 leaks). **Dev paired
  vs the same-scaffold base arm: +1/−3/=36, net −2** (LoRA run
  `20260708T181500-v002-completion-contract` vs base `20260708T132147-…`) —
  below the +5 keep gate, and this was the IN-DISTRIBUTION case (trained on
  these dev tasks' own passing trajectories). Mechanism: the clone reinforced
  the **under-action** failure mode (empty-diff regressions, multi-hundred-turn
  thrashing) rather than fixing it — SFT on the policy's own successes can't
  penalize under-action; that failure wants RL, not imitation. LoRA NOT kept;
  holdout not unlocked.
- **Cycle complete → kill criterion at trigger; scaffold axis CLOSED.** One
  full scaffold+LoRA alternation cycle is done: scaffold +3 (P3), LoRA −2
  (P4), both below +5, no kept variant (v001-baseline is the reference). The
  one pre-registered exception to the stopping rule — **P5 v005-script-first**
  (interaction-geometry hypothesis) — failed its manipulation check NO-GO on
  all three pre-committed criteria (run
  `20260709T014057-v005-script-first-partial`, $0.20): scripts adopted as
  ritual, 9/12 scratch-only diffs vs v001's 9/20 no-source-edit on the same
  tasks. Four intervention classes (prose/forcing/imitation/geometry), one
  invariant failure mode (under-action). Per the 2026-07-09 fork memo
  (FINDINGS) the sanctioned next step is the **$10 best-of-k /
  self-verification precondition test** (evidence: pass@5 = 29/40 = 72.5% vs
  majority-solve 20/40 on run `20260707T215242-v001-baseline` — +9 tasks of
  selection headroom); kill <+5 → negative-result write-up, pass ≥+5 → new
  outside evidence for the RL rung. **Operator decision point.**
- **G6 — not started.** Blocked on the operator decision above.

### Sequencing decision (2026-07-09, operator-directed; supersedes 2026-07-08)

Hosted-Claude reference baselines are **removed, not deferred**. The project's
question is now entirely internal: *does our improvement beat stock Ornith
inside Claude Code on the paired dev/holdout gate?* The external-bar question
(*did we clear the bar that justifies self-hosting vs a hosted model?*) is out
of scope for this project; the hosted-worker code path has been deleted from
the harness so it cannot be run by accident. The v001 Ornith baseline (run
`20260707T215242-v001-baseline`) is the single reference all deltas pair
against. (The earlier 2026-07-08 decision had merely deferred the hosted
columns to pre-G6; this replaces it.)

## Decision rules (pre-committed — surface, don't re-litigate)

- **Minimum detectable difference:** at this corpus size (~40 tasks × 5
  trials) treat paired improvements <5 points as noise. No variant is kept
  below that; a "trend" is not a keep.
- **Stopping rule (Phase 3):** stop the search after 3 consecutive
  non-keeping mutations, then run the holdout gate.
- **Kill criterion (restated 2026-07-09, base-vs-tuned):** if after one
  scaffold+LoRA alternation cycle the improved system does not beat stock
  Ornith + Claude Code by ≥+5 paired tasks on dev (confirmed on holdout once
  staged), the project ships as a negative result + write-up. No extension
  without new outside evidence. *Status: AT TRIGGER — the cycle is complete
  with no kept variant; the $10 best-of-k precondition test (fork memo,
  FINDINGS 2026-07-09) is the sanctioned evidence probe.*
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
  own-repo (private projects with strong test suites), optionally
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
- **P2 Baselines.** Ornith v001 on dev is the reference all deltas pair
  against; full trajectory logging from run one (it is future training data).
  Gate G3 closes on the Ornith baseline entry. Hosted reference baselines
  were removed by the 2026-07-09 redirect (see §Status sequencing decision) —
  all comparisons are internal, base-vs-tuned.
- **P3 Scaffold search (2–3 weeks; most expected gains) — the immediate next
  priority.** Loop:
  run-eval → classify-failures → propose-mutation → operator keeps/rejects.
  Highest-leverage axis: degree of worker self-direction (Ornith's weights
  expect to write their own inner loop). Operator remains the selection step
  for at least the first two cycles; promote to autonomous only after the
  guardrails have caught a real mistake.
- **P4 LoRA (conditional on format-class residue).** **bf16** LoRA (NOT
  QLoRA — Unsloth advises against 4-bit for this Qwen3.5-MoE hybrid) on
  passing trajectories under the winning scaffold; `<think>` preserved in
  targets; rank 32 / α 64; attention only (q/k/v/o_proj), router + GDN/Mamba
  mixer excluded. Re-gate smoke + dev + holdout. **RAN 2026-07-08 — net −2 on
  dev, not kept; see §Status / FINDINGS.**
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
