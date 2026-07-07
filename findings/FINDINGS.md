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
