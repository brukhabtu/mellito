# Findings log

Append-only. This file is the project's memory: paste it (or its tail) into
fresh Claude sessions to restore context. Every entry cites run IDs.

## Entry schema

```
## 2026-07-XX · <phase> · <entry-type: run | mutation | admission | incident | decision>
- variant: vNNN (parent vNNN) — hypothesis: "..."
- run: <run_id> · tasks: N dev · trials: T
- result: pass 62.5% ±4.1 (paired vs parent: +6 tasks / -1 / 31 tie) · $/solved: 0.41 · s/task: 312
- by provenance: public 68% · own-repo 55% · (divergence note if any)
- verdict: kept | rejected | inconclusive — reason
- notes: anything surprising, one line each
```

Decision-rule states (from PLAN.md) to evaluate at each cycle end:
- [ ] Minimum detectable difference respected (no keeps below +5 paired)
- [ ] Dev/holdout gap check (G4 only)
- [ ] Kill criterion: evaluated? met?

---

## 2026-07-07 · P0 Serving · incident (BLOCKED)
- goal: advance frontier gate **G1 Serving** — `modal run infra/modal_app.py::smoke`
  exits 0 (schema-clean tool calls, no `<think>` leakage, 20/20 trivials).
- frontier evidence: `python3 infra/status.py` reports `Frontier: G1`; G1 line
  `[----] smoke suite exits 0  (modal CLI missing)`.
- G1 decomposition (goal skill) and per-criterion status:
  - **U1 Modal account + auth** — criterion: `modal token` resolves a valid
    account. RESULT: unsatisfiable. No `modal` CLI, no `modal` Python package,
    no `~/.modal.toml`, no `~/.config/modal`, no `MODAL_TOKEN_*` in env.
  - **U2 Gated HF weights access** — criterion: `deepreinforce-ai/Ornith-1.0-35B-FP8`
    is pullable. RESULT: unsatisfiable. Model is gated; no HF token/cache present.
  - **U3 vLLM endpoint deployed** (`modal deploy`, H100, patched chat template).
    RESULT: unreachable — depends on U1+U2; also GPU-spend-bearing under the
    $150 cap in infra/budget.yaml.
  - **U4 smoke() body implemented** — currently `raise SystemExit("smoke: not
    implemented (Phase 0)")`. RESULT: implementable in-repo, but **moot**: even a
    complete smoke suite cannot exit 0 without a live endpoint (U1–U3). Writing
    it now would ship serving code that cannot be exercised → an unverified
    number, which experiment-integrity forbids. Deferred until the serving
    stack is authenticatable.
- verdict: **BLOCKED — stop condition (c): missing credentials/accounts (Modal
  auth, HF weights access).** Binding constraint is U1 (and U2). This is the
  expected first block per the milestone plan.
- action: no workflow fleet dispatched — every worker's first mechanical step
  (`modal token` / `modal deploy`) would fail identically; a fan-out here is
  thrash, not progress. No credentials fabricated or injected; no guard hook
  routed around.
- unblock path (operator): provide a Modal account (`modal token new` or
  `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` as a Modal secret / env), accept the
  Ornith-1.0-35B-FP8 license and provide an `HF_TOKEN`, then re-run
  `infra/status.py`. Frontier advances to U3→U4 (implement + deploy) once auth
  resolves.
- decision-rule states this cycle: MDD n/a (no runs) · dev/holdout gap n/a
  (G4 only) · kill criterion n/a (no cost/pass data yet).

## 2026-07-07 · P0 Serving · incident (partial unblock — still BLOCKED)
- update to the G1 block above: operator completed Modal web auth.
- evidence: `modal profile current` → `bruk-habtu`; token at `~/.modal.toml`.
  `python3 infra/status.py` G1 line now reads `[----] smoke suite exits 0
  (exit 1)` — the CLI resolves, authenticates, and actually executes
  `modal run infra/modal_app.py::smoke`, which exits 1 from its unimplemented
  Phase-0 stub. Prior state was `(modal CLI missing)`.
- unit status now: **U1 Modal auth RESOLVED**. U2 gated HF weights still
  unsatisfied (`modal secret list` empty; no `HF_TOKEN` in env or ~/.cache).
  U3 deploy + U4 smoke-body remain, gated on U2.
- verdict: **still BLOCKED — stop condition (c)**, binding constraint now
  narrowed to U2 (HF gated-weights access). No GPU spent; budget ledger empty.
- unblock path (operator): accept the `deepreinforce-ai/Ornith-1.0-35B-FP8`
  license on Hugging Face and provide an `HF_TOKEN`; it will be wired as a
  Modal secret (`modal secret create huggingface HF_TOKEN=...`), never
  committed. Then: implement U4, `modal deploy`, re-run status.py.

## 2026-07-07 · P0 Serving · incident (surprise — U2 was not a blocker)
- surprise (logged per experiment-integrity before acting): the weights model
  `deepreinforce-ai/Ornith-1.0-35B-FP8` is **public, not gated**. Evidence:
  `GET https://huggingface.co/api/models/deepreinforce-ai/Ornith-1.0-35B-FP8`
  returns HTTP 200 unauthenticated with `gated:false, private:false`. Operator
  reported the model page shows no license/access prompt — consistent.
- correction: the prior two entries treated U2 (gated HF access) as the binding
  blocker. It is not. No license acceptance is required; weights download
  anonymously. An `HF_TOKEN` remains optional (rate-limit insurance for the
  ~35GB pull inside Modal), not a gate.
- net G1 unit status: **U1 Modal auth RESOLVED · U2 weights access N/A (public)**.
  Remaining is pure implementation: U4 (smoke body) + U3 (`modal deploy`,
  H100 GPU spend under the $150 cap; ledger currently empty). G1 is now
  actionable, no missing credentials.

## 2026-07-07 · P0 Serving · incident (serving surprise, logged before workaround)
- context: first `modal deploy` + cold boot of Ornith on 1×H100, vLLM 0.24.0.
- confirmed good: vLLM resolves `Qwen3_5MoeForConditionalGeneration`, accepts
  `--tool-call-parser qwen3_xml` / `--reasoning-parser qwen3` / `max_model_len
  32768`; Mamba/GDN hybrid handled ('align' cache mode); weights are public and
  pulled anonymously (35.09 GiB in 92 s via Xet, now cached in the volume — a
  few transient `Connection reset by peer` on file-listing, auto-retried).
- surprise / blocker: the model's **gated-delta-net linear-attention** layers
  use FlashInfer's GDN prefill kernel, which is **JIT-compiled and needs
  `nvcc`**. The `debian_slim`+pip vLLM image has no CUDA toolkit →
  `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
  doesn't exist`, repeated → EngineCore restart loop (pid 74→73), never binds
  :8000. vLLM's own log names the fix: "Set --gdn-prefill-backend triton".
- workaround (applied next commit): add `--gdn-prefill-backend triton` (Triton
  JIT needs no nvcc) and `--enforce-eager` (skip torch.compile + cudagraph
  capture over 51 sizes) for a fast, robust **G1 boot**. Correctness is
  unchanged; both are throughput/latency knobs to revisit in P2 (either bake
  nvcc + re-enable compile for perf, or keep eager if fast enough).
- cost note: no ledger entry (smoke is a gate, not a sweep); GPU time was the
  cold-boot loop only. status.py still at frontier G1.

## 2026-07-07 · P0 Serving · decision (G1 CLOSED — gate met)
- goal: G1 Serving — Ornith-1.0-35B-FP8 on Modal behind an OpenAI/Anthropic-
  compatible endpoint; `modal run infra/modal_app.py::smoke` exits 0.
- pinned criteria & results:
  1. endpoint serves the model, `/health` 200 — **PASS** (deployed serve,
     `smoke: endpoint healthy`).
  2. `smoke` exits 0 — **PASS** (exit code 0).
  3. 20/20 trivial deterministic tasks correct — **PASS** (`trivials 20/20`).
  4. no `<think>` leakage in any content — **PASS** (`think-leaks 0`; qwen3
     reasoning parser strips reasoning into a separate field).
  5. schema-clean tool call → valid Anthropic tool_use — **PASS**
     (`tool-call schema OK`; qwen3_xml parser).
- gate evidence: `python3 infra/status.py` → `G1 Serving [PASS] smoke suite
  exits 0 (exit 0)` · `Frontier: G2`. (status.py re-runs smoke itself, so the
  PASS is a fresh cold-path validation, not a cached claim.)
- serving config that works (1×H100, vLLM 0.24.0): arch
  Qwen3_5MoeForConditionalGeneration; `--max-model-len 32768`,
  `--enforce-eager`, `--gdn-prefill-backend triton`,
  `VLLM_USE_FLASHINFER_SAMPLER=0`, qwen3_xml tool parser, qwen3 reasoning
  parser, prefix caching, bundled chat template. Two nvcc-JIT blockers found
  and fixed (GDN prefill kernel; FlashInfer sampler) — both logged above.
- observations for later phases: generation throughput is low (~10 tok/s in
  eager + triton-GDN) — a P2 tuning target (bake nvcc + re-enable
  compile/cudagraph, or measure whether eager is acceptable), not a G1 blocker.
  Weights (35 GiB) now cached in the `ornith-weights` volume; warm boots skip
  the download.
- orchestration note: G1 was executed **in-session**, not via a fan-out
  Workflow. Its units are coupled (single serving file) and culminated in one
  supervised, iterative live-deploy debug loop (three deploy→diagnose→redeploy
  cycles chasing nvcc/JIT issues) — not parallelizable and requiring live
  supervision, so a worker fleet would add overhead, not coverage. The reusable
  `.claude/workflows/milestone.js` wave orchestrator will be authored and first
  exercised at **G2 (corpus)**, which is genuinely fan-out shaped (≥40 dev +
  ≥15 holdout independent hermetic-task admissions through corpus-curator).
- decision-rule states this cycle: MDD n/a (no variant comparison yet) ·
  dev/holdout gap n/a (G4 only) · kill criterion n/a (no cost/pass data yet).
