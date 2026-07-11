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

## 2026-07-07 · P1 Corpus · incident (holdout guard block — data point)
- frontier is now **G2 Corpus** (status.py). While probing G2 prerequisites, a
  Bash command that listed `tasks/holdout` was BLOCKED by guard-holdout.py
  (exit 2). Recorded as a data point per experiment-integrity; not worked
  around. Re-ran the probe excluding the sealed path.
- G2 scoping facts gathered: Docker present (v29.3.1) so the 3+3 determinism
  check can run locally; R2E-Gym and SWE-Gym dataset APIs reachable (HTTP 200);
  `run_trial`/`run_sweep` still Phase-1 TODO stubs; tasks/dev and tasks/staging
  empty; own private repos not in this session's scope.
- G2 critical path (to be run as the milestone.js wave once sourcing is
  decided): import/author ≥40 dev hermetic tasks (pinned image + binary
  verify) → 6/6 determinism per task → stage ≥15 holdout-destined
  (own-repo/post-cutoff only) for the operator to move. Holdout population is
  operator-gated by design (curator item 4 + the hook).

## 2026-07-07 · P1 Corpus · run (determinism harness built + validated)
- unit: the 3+3 determinism admission check (corpus-curator item 2), as
  reusable infra `infra/determinism_check.py` — Docker-based, source-agnostic
  (SWE-bench/R2E-Gym/SWE-Gym/own-repo all express broken vs solution states as
  shell steps in a JSON descriptor), binary offline verdict path, `--network
  none` during verify.
- pinned criteria & results:
  1. admits a hermetic task (broken fails 3/3, solution passes 3/3) — **PASS**
     (`infra/tests/determinism_selftest.json` → `6/6`, admissible, exit 0).
  2. rejects a non-hermetic task (broken state doesn't fail) — **PASS**
     (negative self-test → `3/6`, admissible=false, exit 1).
  3. identical-base guarantee: solution state is built from a snapshot of the
     post-setup base, so it differs from broken only by the gold fix — **PASS**
     (implemented via `docker commit` of the base, re-used for both states).
  4. self-test fixtures live in `infra/tests/`, NOT `tasks/dev/` — **PASS**
     (no corpus-count inflation; status.py dev count still 0).
- environment recon: Docker daemon started locally (v29.3.1, overlayfs, root);
  usable disk ~30 GiB (quota-limited); the standard SWE-bench prebuilt eval
  image for sampled SWE-Gym instances (e.g. getmoto__moto-7365) is **absent**
  on Docker Hub — SWE-Gym imports need SWE-Gym's own image-build harness, not a
  prebuilt pull.
- verdict: G2 **not closeable this session** — the machinery is ready, but
  admitting real tasks is gated on three operator-side inputs, none a code fix:
  (a) dev sourcing given no prebuilt images (build SWE-Gym env images — heavy,
  disk-bound at 30 GiB — vs run determinism on Modal vs own-repo tasks);
  (b) own-repo access (private repos not in this session — need add_repo);
  (c) holdout population (own-repo/post-cutoff only; I stage, operator moves).
  Recorded as the current frontier state; awaiting operator direction rather
  than thrashing a blind 40-image build or inflating the count with non-admitted
  tasks (integrity: numbers/tasks must be real).
- decision-rule states this cycle: MDD n/a · dev/holdout gap n/a (G4 only) ·
  kill criterion n/a (no cost/pass data yet).

## 2026-07-07 · P1 Corpus · admission (dev set built to 40 — G2 dev criterion MET)
- goal: advance **G2 Corpus** — the dev half (≥40 hermetic tasks, each 3+3
  determinism, provenance recorded).
- correction to prior "no prebuilt images" block: that was specific to
  **SWE-Gym**. **SWE-bench Verified** ships a prebuilt per-instance image for
  every task on Docker Hub (`swebench/sweb.eval.x86_64.<id>`, `__`→`_1776_`),
  with the repo at `base_commit` under `/testbed` and a `testbed` conda env
  holding deps. So dev sourcing needs **no image build** — each instance is a
  ready-made hermetic task. Verified: pulled `psf_1776_requests-1142`, ran the
  full 3+3 → 6/6. Crucially, the determinism check needs only ONE image
  resident at a time (pull → check → admit → `docker rmi`), so the ~30 GiB disk
  quota is not a cap on corpus size.
- machinery built: `infra/import_swebench.py` (`descriptor` | `admit` | `batch`
  | `list`). It maps a Verified instance onto the determinism descriptor:
  base = image `/testbed` hard-reset to `base_commit`; to_broken = apply
  `test_patch`; to_solution = apply gold `patch`; verify = the repo's real
  SWE-bench test runner (django `runtests.py`, others `pytest`, etc.) via
  `swebench`'s `MAP_REPO_VERSION_TO_SPECS[test_cmd]` + `get_test_directives`,
  under the `testbed` env, exit 0 = pass. `determinism_check.py` unchanged.
- results: **40/40 dev tasks admitted, every one 6/6** (django 28, astropy 6,
  pylint 3, scikit-learn 2, pytest 1). `python3 infra/status.py` →
  `G2 Corpus [PASS] dev tasks >= 40 (40)`. Provenance on all = `public-pretrained`
  (honest: every Verified repo is public and predates the 2026-06-25 cutoff).
- 4 conservative REJECTS (correct, not failures): pylint-4661, requests-1142,
  scikit-learn-14053, astropy-8707 — all 3/6. Cause: SWE-bench's directive
  scope runs whole test files/modules; a neighbour test that needs network
  (e.g. requests' httpbin) or is otherwise offline-flaky makes the solution
  state fail 3/3 too → not 6/6. The check correctly refused them; the
  over-provisioned candidate pool (66) absorbed the loss.
- reproducibility guard: re-ran determinism on `django-15127` from the **pinned
  digest** recorded in its `task.yaml` (not the tag) → 6/6. Corpus is
  reproducible downstream from the digests, not a cached fluke.
- infra notes (surprises, logged before working around): (1) Docker Hub
  503-throttled the daemon's **direct** egress after ~a dozen pulls; fix was to
  route `dockerd` through the agent proxy (`HTTPS_PROXY=127.0.0.1:43719`) and
  trust the proxy CA (`update-ca-certificates`) — pulls then stable. (2)
  Installed `swebench==4.1.0` for the per-repo test-command specs. (3) tox-based
  repos (sphinx) excluded from candidates: `tox` wants to (re)build a venv,
  which fights `--network none`; an earlier buggy pass false-admitted
  sphinx-8595 under a naive `pytest` command — caught and wiped in the clean
  rebuild.
- **caveats surfaced for operator (not gate failures):**
  1. **Repo skew:** 28/40 (70%) are django. A scaffold could overfit django's
     test idioms. The non-django small-fast-test pool in Verified is thin;
     rebalancing = one more `batch` run with lower django caps + higher
     astropy/sklearn/xarray/matplotlib caps (costs ~more pulls of slower suites).
  2. **Dev is 100% public-pretrained** — the PLAN's corpus strategy wants
     ~half own-repo (private) in dev. Own repos are **not in this
     session's scope** and `list_repos` is unavailable for account-owned
     sessions, so own-repo dev tasks are deferred pending an operator
     `add_repo`. Memorization inflates all variants ~equally under paired
     comparison, so a public-only dev set is usable for the search; the
     contamination tripwire relies on the own-repo **holdout** slice.
- decision-rule states this cycle: MDD n/a (no variant runs) · dev/holdout gap
  n/a (G4 only) · kill criterion n/a (no cost/pass data yet).

## 2026-07-07 · P1 Corpus · incident (holdout BLOCKED — SWE-rebench post-cutoff assumption is stale)
- G2's **holdout** criterion (≥15, own-repo/post-cutoff only) is unmet (0/15)
  and cannot be sourced from public benchmarks this session. Investigated the
  two legal routes:
  - **SWE-rebench** (`nebius/SWE-rebench`, public): 21,336 tasks, schema carries
    `docker_image`/`created_at`/FAIL_TO_PASS. But **latest `created_at` is
    2025-04-30**; **zero** tasks ≥ the 2026-06-25 cutoff. The PLAN's assumption
    that SWE-rebench would supply freshly-mined post-cutoff holdout is **stale**
    — the published dataset hasn't been updated past 2025-04. So it yields no
    legal (post-cutoff) holdout under the plan's own definition.
  - **Own-repo reserved bugs** (private): the only remaining legal source,
    but those repos are not in this session and `list_repos` is unavailable for
    account-owned sessions.
- verdict: **holdout BLOCKED — needs operator input**, binding constraint is
  own-repo access. Unblock path: operator `add_repo`s a private repo; I
  then author reserved-bug tasks (pinned image + binary verify), 3+3 each, and
  **stage** them in `tasks/staging/` — the operator moves them into
  `tasks/holdout/` (guard-holdout.py blocks me by design). No holdout tasks
  fabricated; no public-pretrained task mislabelled as post-cutoff.
- decision-rule states this cycle: MDD n/a · dev/holdout gap n/a · kill n/a.

## 2026-07-07 · P1 Corpus · admission (dev rebalanced for repo diversity)
- operator called the 70% django skew; rebalanced. Dropped 14 django (kept 14),
  admitted 14 more non-django via the same `batch` path against a non-django
  candidate pool. Final dev (40): django 14 · scikit-learn 8 · astropy 6 ·
  pylint 6 · pytest 4 · xarray 2. Max single-repo share now 35% (was 70%).
  Every task still 6/6; `status.py` G2 dev still PASS (40).
- same conservative rejects recurred (requests-1142, pylint-4661,
  sklearn-14053/-14053, astropy-13033/-8707) — whole-file directive scope pulls
  an offline-flaky/network neighbour test; correct refusals.

## 2026-07-07 · P1 Corpus · incident (no public source yields post-cutoff holdout)
- operator ruled out own-repo holdout and asked for a **public** holdout
  source. Researched all
  public options; **none supplies post-2026-06-25 tasks**:
  - **SWE-rebench** (`nebius/SWE-rebench`): 21,336 tasks, latest 2025-04-30.
  - **SWE-bench-Live** (`SWE-bench-Live/SWE-bench-Live`): monthly splits, latest
    is `202506` (2025-06-20), 50 tasks/month, repos disjoint from dev (mypy,
    dspy, textual, fastmcp, …). But rows carry **no prebuilt image** (only
    `test_cmds`/`base_commit`), so tasks need image-building via SWE-bench-Live's
    harness — not a turnkey pull like SWE-bench Verified.
  - SWE-bench Verified/Lite: all pre-cutoff public-pretrained (already dev).
- conclusion: a **genuine post-cutoff public holdout requires mining GitHub PRs
  merged after 2026-06-25 and building hermetic images ourselves** — no dataset
  has tasks that fresh. This is a distinct sub-project (env-image build per task,
  no prebuilts). Two framings for the operator to choose between:
  (A) **strict** — mine fresh (post-2026-06-25) bugfix PRs from easy-to-build
      public repos; genuine uncontaminated holdout; heavier, higher build-failure
      rate.
  (B) **best-effort** — use SWE-bench-Live `202506` tasks (2025-06, repos
      disjoint from dev) as a held-out-*repos* proxy; faster (reuse importer once
      images build) but weaker contamination guarantee — must be logged as such,
      and the contamination tripwire's meaning weakens.
- no holdout tasks fabricated; no pre-cutoff task mislabelled post-cutoff.
  G2 holdout remains 0/15 pending the operator's framing choice.

## 2026-07-07 · P1 Corpus · admission (holdout staged — 18 best-effort tasks; operator move pending)
- operator chose the **best-effort** holdout framing (SWE-bench-Live 2025-06).
  Discovery that made it tractable: SWE-bench-Live ships prebuilt per-instance
  images under the **`starryzhang/sweb.eval.x86_64.<id>`** namespace (repo at
  `/testbed`, system python, no conda) — same pull-and-check pipeline as dev,
  no image build needed. (The dataset parquet's image fields are null; the
  naming is deterministic.)
- machinery: `infra/import_swebench_live.py` (`descriptor`|`batch`|`list`).
  Verify is built from each row's `test_cmds` (all `log_parser=pytest`): keep the
  env wrapper (uv/poetry/`python -m`/PYTHONPATH) but run exactly the
  FAIL_TO_PASS node ids with `-rA`, exit 0 = pass. Excludes any repo present in
  dev (disjoint-repos invariant). Writes to `tasks/staging/` only.
- result: **18 tasks staged, every one 6/6**, across 15 distinct repos (agentops,
  litellm, aider, textual, praisonai, pyomo, cfn-lint, aiogram, beets, briefcase,
  powertools-lambda, django-celery-beat, checkov, certbot ×2, conan ×2) — none
  overlapping the dev repos. Dates 2025-04…2025-06. 1 reject (textual-5823, 3/6,
  offline-flaky neighbour). Reproducibility re-checked on agentops-1002 from its
  **pinned digest** → 6/6.
- provenance: labelled **`held-out-public`** (new, documented in tasks/schema.md
  + .claude/rules/tasks.md): public repos disjoint from dev, ~2025-06, NOT strict
  post-cutoff. Honest about the weaker contamination guarantee — a dev/holdout
  gap here is a **generalization** signal (unseen repos), not a clean
  contamination verdict. This is the operator-approved deviation from
  "holdout = own-repo/post-cutoff".
- **operator action required to close G2 holdout:** move ≥15 of the 18 staged
  specs from `tasks/staging/` into `tasks/holdout/` (I cannot — guard-holdout.py
  blocks writes there by design; that block is correct). After the move,
  `status.py` G2 holdout flips to PASS and the frontier advances to G3.
- caveat carried forward to G3/G4: baseline + search results must be sliced
  public-pretrained (dev) vs held-out-public (holdout); the contamination
  tripwire's strength is reduced accordingly. If a genuine post-cutoff holdout
  is wanted later, the fresh-PR-mining route (strict option) remains open.
- decision-rule states this cycle: MDD n/a · dev/holdout gap n/a (G4 only) ·
  kill criterion n/a (no cost/pass data yet).

## 2026-07-07 · P1 Corpus · incident + fix (corpus was not eval-runnable; hidden tests now persisted)
- session context: fresh ephemeral container. Two environment regressions logged
  as data points (not worked around): (1) **G1 regressed to frontier** —
  `status.py` re-runs `modal run ...::smoke` live and it exits 1 because the vLLM
  endpoint isn't deployed in this container (the G1 PASS was a prior session's
  live deploy; deploys don't persist). (2) **Modal auth absent** — no
  `~/.modal.toml`, `modal profile current` = default. So no serving, no sweeps,
  no baselines this session without an operator re-auth. (3) A probe that listed
  the sealed holdout split was **BLOCKED by guard-holdout.py** (exit 2) — correct;
  recorded, not circumvented. Holdout still 0/15 (operator move of the 18 staged
  specs pending).
- **gap found in G2 output:** the dev/staging `task.yaml` stored only the F2P
  `verify` command, NOT the instance's `test_patch`. The determinism check
  applied the tests itself (via the descriptor's to_broken), so admission passed
  — but a real eval gives the worker the repo at `base_commit` with NO tests
  visible, then must inject the hidden tests only at verdict time. Without the
  stored tests, `verify` references tests absent from the base image → the
  corpus was **determinism-clean yet not eval-runnable**. Confirmed on
  django-10973: bare base + verify errored; gold-fix + tests -> PASS,
  tests-only -> FAIL.
- **fix:** `import_swebench.py patches` / `import_swebench_live.py patches`
  persist each admitted task's `test_patch` as `tests.patch` beside its
  task.yaml (from the datasets, no image re-pull) and add `hidden_tests:
  tests.patch`. Applied to all **40 dev + 18 staged**. Verified end-to-end with
  the STORED tests: worker-fixed+tests -> PASS, no-op+tests -> FAIL. Documented
  the hidden-tests eval-verdict contract in tasks/schema.md and encoded it in
  `run_trial`'s skeleton (tests injected at verdict time only, never in the
  worker workspace — no oracle leakage).
- net: G2 dev is now genuinely eval-ready (40 self-contained tasks). Frontier
  remains operator-gated: **re-auth Modal** (restores G1 serving + unblocks G3
  sweeps/baselines under the $150 cap) and **move >=15 staged specs into the
  holdout split** (closes G2). G3 runner bodies (run_trial/run_sweep) stay
  Phase-1 TODO — deliberately not written blind, since they can't be exercised
  without a live endpoint (same integrity stance as the G1 smoke-body deferral).
- decision-rule states this cycle: MDD n/a · dev/holdout gap n/a (G4 only) ·
  kill criterion n/a (no cost/pass data yet).

## 2026-07-07 · P0 Serving · run (G1 re-confirmed live after container reset)
- context: the ornith-harness `serve` app was still deployed from the prior
  session (`modal app list` shows it). After re-auth (see modal-auth skill),
  ran `modal run infra/modal_app.py::smoke` against it.
- flake (logged per experiment-integrity before re-running): first run scored
  **19/20** — trivial 5 failed with `HTTP 500 Internal Server Error`, a
  transient server-side error (think-leaks 0, tool-call schema OK; the other 19
  correct). Endpoint logs showed all other requests 200 OK at ~13 tok/s.
- re-run on the warm endpoint: **smoke PASS — 20/20 trivials, 0 think-leaks,
  schema-clean tool call** (run `ap-foZEfhxyx3EJ1mMgKWaajg`). The 500 did not
  recur; treated as a transient endpoint blip, not a regression.
- **implication for G3 run_trial:** the endpoint can emit a transient 5xx under
  load. Per PLAN "Error ≠ fail", a trial hitting a transient 5xx must be retried
  a bounded number of times and, if still failing, recorded as verdict=`invalid`
  (excluded from paired stats), never `fail`. Bake this into run_trial's worker
  loop and verdict path.
- serving perf unchanged: ~13 tok/s generation (eager + triton-GDN). A full dev
  sweep (40 tasks × trials × agentic multi-step) will be slow — size trial
  timeouts and budget accordingly; P2 throughput tuning (bake nvcc, re-enable
  compile/cudagraph) remains a later lever.

## 2026-07-07 · P2 Measurement · run (G3 machinery built + tested; worker body throughput-blocked)
- goal: G3 measurement machinery. Pinned criteria:
  1. sweep aggregation (per-task pass, paired win/loss/tie + net vs parent,
     provenance slices, Wilson CI, cost/$-per-solved) — pure & unit-tested. **PASS**
     (`infra/sweep_stats.py`; synthetic 3-task test: solved 2/3, paired
     +1/-1/=1 net 0, invalid excluded, ledger format correct).
  2. `run_sweep` wired end-to-end: load specs + inline `hidden_tests`, tar the
     variant's claude-config, fan out run_trial, aggregate, write
     `experiments/runs/<run_id>/summary.json` + append cost-ledger.csv; refuses
     holdout without `.holdout-unlocked`; refuses trials<3. **PASS** (validated
     locally on real dev specs, minus the Modal starmap: spec+tests load, config
     tar 559 B, summary + ledger written).
  3. run_trial verdict path (hidden-tests contract) specified + proven locally
     earlier (gold+tests->PASS, noop+tests->FAIL). **PASS**
  4. no unverified numbers: all checks mechanical, no GPU claimed. **PASS**
- **out of scope / BLOCKED — run_trial worker body:** the agentic worker loop
  (modal.Sandbox on the task image + Claude Code CLI against Ornith via an
  Anthropic-compat proxy + token/gpu accounting) is NOT built. Deliberately
  deferred, because it is **throughput-blocked**:
  - measured serving throughput is ~13 tok/s (eager + triton-GDN, G1 config).
  - an agentic SWE trajectory is many model calls; even a modest 20k output
    tok/trial = ~26 min/trial. A 40×5 Ornith baseline = ~85 GPU-hours ≈ **$342**,
    over the **$150** cap (guard-budget would block it). Even an optimistic 5k
    tok/trial run is ~$85 for one model.
  - conclusion: a real G3 baseline is not feasible at the G1 boot config.
    **P2 throughput tuning is now a prerequisite, not a later lever** — bake a
    CUDA toolkit (nvcc) into the serve image and re-enable torch.compile +
    cudagraph capture (the two nvcc/JIT blockers were the reason for
    `--enforce-eager` + triton-GDN at G1), and/or raise request concurrency, to
    lift tok/s enough that trials are minutes not hours.
- decision-rule states: MDD n/a (no runs) · dev/holdout gap n/a · kill n/a.
- run ledger: none yet (no GPU-bearing sweep executed — machinery only).

## 2026-07-07 · P2 Throughput · incident (batching gives NO benefit; ~13 tok/s ceiling confirmed)
- probed the live endpoint's aggregate throughput vs concurrency (to test whether
  the parallel sweep amortises the ~13 tok/s single-stream latency into higher
  aggregate throughput — which would make the G3 baseline cheap).
- result (warm endpoint, 256-tok completions):
  - concurrency=1: ~13 tok/s (the one 1.1 tok/s reading was cold-boot-inflated).
  - concurrency=8: **2048 tok in 158s = 13.0 tok/s AGGREGATE** — identical to
    single-stream. Per-request latencies were **staggered 21s→158s**, not
    clustered, i.e. requests were served **serially, not batched**.
- conclusion: the endpoint has a hard ~13 tok/s aggregate ceiling that does NOT
  scale with concurrency. The continuous-batching hope that would have made the
  sweep cheap is dead — the earlier ~$342 estimate for a 40×5 Ornith baseline
  stands (or worse, given no batch amortisation). **G3 baselines are infeasible
  at the current serve config; P2 throughput tuning is a hard prerequisite.**
- likely causes to chase in P2 (staggered latencies = serialization):
  1. `--enforce-eager` (no cudagraph) — per-step overhead dominates; re-enabling
     compile+cudagraph needs a CUDA toolkit (nvcc) baked into the serve image
     (the reason it was disabled at G1).
  2. hybrid GDN/Mamba state batching limits in vLLM 0.24 (check `--max-num-seqs`,
     `--max-num-batched-tokens`, and whether the scheduler runs >1 seq at once).
  3. Modal serve container request concurrency (`@modal.concurrent`) — confirm
     vLLM actually receives the 8 requests simultaneously vs Modal queuing them.
- no ledger entry (probe only; ~1 H100 cold boot + a few short completions).
  Endpoint scales down after 120s idle.
- decision-rule states: MDD n/a · dev/holdout gap n/a · kill n/a (no runs).

## 2026-07-07 · P2 Throughput · run (Modal concurrency fix: 13 -> 28 tok/s aggregate)
- root cause of the flat ~13 tok/s: the serve `@modal.web_server` had no
  concurrent-input setting, so Modal routed ONE request per container at a time
  — vLLM's engine logs showed `Running: 1 reqs, Waiting: 0` throughout an
  8-client probe. vLLM's continuous batcher never saw a queue. NOT a model limit.
- fix: added `@modal.concurrent(max_inputs=64)` to `serve()` (vLLM command
  unchanged), redeployed.
- re-probe (warm): c=1 15.6 tok/s · **c=8 28.0 tok/s aggregate** (was 13.0) — a
  2× lift from real batching. Per-request latencies still staggered (16–73s), so
  batching is partial, not the full 8× — more P2 headroom remains (candidates:
  raise `--max-num-seqs`, drop `--enforce-eager` for cudagraph, check hybrid
  GDN/Mamba batch limits, push max_inputs/target_inputs).
- cost impact: roughly halves the earlier estimate — a 40×5 Ornith baseline
  ~$170 at 28 tok/s (still near the $150 cap; further tuning or a reduced first
  baseline needed). Iteration latency also halved.
- G1 unaffected by design (serve command identical); a smoke re-confirm is
  advisable before relying on it.
- decision-rule states: MDD n/a · dev/holdout gap n/a · kill n/a.

## 2026-07-07 · P2 Throughput · run (cudagraph+compile: 13 -> 908 tok/s aggregate — G3 unblocked)
- replaced `--enforce-eager` with `--compilation-config
  '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}'` (torch.compile inductor +
  piecewise/full cudagraph). vLLM confirmed `enforce_eager=False`,
  `cudagraph_mode=FULL_AND_PIECEWISE`, compiled a graph in ~48s at boot; the
  GDN attention is in `splitting_ops` so no nvcc path is hit (triton GDN +
  native sampler workarounds retained). Boot is slower (one-time compile) but
  clean — no crash loop.
- re-probe (warm, 256-tok completions):
  - c=8: 2048 tok in **4.0s = 518 tok/s** aggregate (per-req all ~3.9s — truly
    batched, not staggered).
  - c=24: 6144 tok in **6.8s = 908 tok/s** aggregate (per-req all ~6.6s).
  - vs the eager config's flat 13 tok/s: a **~70× aggregate speedup**. Warm
    single-stream also ~2.5× (≈39 tok/s from the batch-24 per-req rate).
- root cause recap: eager mode's per-decode-step overhead both throttled
  single-stream AND swamped batching (the earlier "Running: 1" symptom). With
  cudagraph the engine batches the whole fleet.
- **cost impact — G3 is now cheap:** at 908 tok/s a 40×5 Ornith baseline is
  ~$1 (20k tok/trial) to ~$3 (60k tok/trial) of H100 time — far under the $150
  cap, and minutes of wall-clock. The throughput blocker is resolved; G3 can run.
- correctness CONFIRMED: G1 smoke re-run on the compiled config = **PASS 20/20
  trivials, 0 think-leaks, tool-call OK** (run ap-RW4x5gYvUMJZb9c0ZwrnmF), and it
  finished in ~90s vs ~10 min under eager. Compiled greedy output is identical.
- @modal.concurrent(max_inputs=64) from the prior fix is retained and is what
  lets the fleet's requests reach the engine together.

## 2026-07-07 · P2 Validation · run (independent validation pass over the PR #2 merge)
- goal (approved plan, Phase A): validate the merged corpus + machinery before
  any paid run. Executor split per operator: Fable ran/adjudicated; a Sonnet
  subagent authored the mechanical tooling.
- results, per pinned criterion:
  1. **smoke re-gate PASS** — 20/20 trivials, 0 think-leaks, schema-clean tool
     call on the compiled config (run `ap-ne7YzYVrAaUKSvjAvlmFiw`).
  2. **corpus audit PASS** — new `infra/audit_tasks.py` (58 specs: 40 dev + 18
     staging; holdout never read): required fields, id/dir match, split-legal
     provenance, digest-pinned images, hidden-tests present + diff-shaped,
     timeout bounds, verify non-empty → **0 failures**.
  3. **oracle-leak adjudication** — the audit's strict substring check first
     flagged 9 tasks; inspection showed all were *repro-snippet overlap*
     (SWE-bench builds its tests from the issue's own repro, so imports/repro
     lines appear in both — issue→test direction, no fix/verdict revealed).
     Hard-FAIL now reserved for hidden-test *names* (`def test_…`) in the
     description → **0 real leaks**; the 9 downgraded to WARNs, logged.
  4. **determinism re-verified from pinned digests** — django__django-10973 and
     pylint-dev__pylint-6386 both **6/6** (fresh dockerd, images re-pulled by
     digest). Corpus remains reproducible downstream.
  5. **stats tests formalized** — `infra/tests/test_sweep_stats.py`, 18 pytest
     cases over per_task/paired/provenance/cost/summary/ledger — **18 passed**.
- code-review of the run_trial/run_sweep surface happens with the Phase B diff
  (one review covers both) before the first paid trial.

## 2026-07-07 · P2 Measurement · run (adversarial review of run_trial before first paid trial)
- per the approved plan, the G3 runner was code-reviewed (Fable, 8 finder
  angles via subagents, 1-vote verify) before any paid run. **10 confirmed
  findings**, all fixed in the same cycle (fix implementation by the Opus
  builder from a pinned fix list; re-reviewed + spot-checked by Fable; 34/34
  tests green). Highest-severity, all verified against modal 1.5.1 source or
  by experiment:
  1. one raising trial aborted the whole sweep AND lost all finished results +
     the ledger row (starmap return_exceptions=False default; trials.jsonl
     written post-loop) → now return_exceptions=True + per-trial invalid.
  2. ALL ExecTimeoutError handling was dead code (modal ends streams silently,
     wait() returns -1) → timeouts were double-billed retries landing as
     invalid; now detected via rc==-1 + elapsed, timeout proceeds to diff.
  3. CLI is_error results fell through to empty_diff → 'fail' (Error≠fail
     violation) → now invalid 'worker_reported_error'.
  4. git diff --binary read through a strict-UTF-8 text stream (crash on
     non-UTF-8 fixtures) → bytes end-to-end.
  5. serve()/proxy() were publicly routable with NO auth and a repo-hardcoded
     proxy key (CLAUDE.md secrets rule) → new `ornith-endpoint-keys` Modal
     secret (VLLM_API_KEY + PROXY_MASTER_KEY, values never in repo/echoed);
     vllm --api-key; proxy master key from env; smoke fetches the key via a
     secret-attached function.
  6. phantom api_usd on ornith trials (Claude Code self-prices tokens even via
     proxy) + base_url used as the ornith/claude switch → explicit
     worker.gpu_attributed drives gpu vs api attribution.
  7. run_one bypassed the RUN_SWEEP_API_OK API-spend ack → gate moved into
     _make_worker (single choke point).
  8. unchecked git/scaffold exit codes booked infra faults as model 'fail'
     (base_sha='', failed materialization, failed git add) → explicit invalid
     reasons.
  9. partial/claude-run exclusion from paired stats relied on run_id string
     luck → summaries now carry partial/worker_model/variant fields and
     _parent_per_task selects on them.
  10. trial_logic.classify verdict table was unwired dead code → run_trial now
     routes every stage→verdict through it (retry orchestration stays in the
     caller).
- also: oracle-leak tar scan moved to _tar_config (fails locally before any
  container is scheduled; in-trial backstop kept), URL resolvers deduped,
  attribution string derives from AGG_TOK_PER_S.
- residual known risks (accepted, logged): timeout detection is time-based
  (rc -1 near deadline from another kill mislabels as timeout); smoke pays one
  extra slim-container start to fetch the endpoint key; legacy summaries
  without the new fields are treated as non-partial ornith.

## 2026-07-07 · P2 Measurement · run (proof-of-one PASS — run_trial validated end-to-end)
- first real Ornith trial through the full run_trial pipeline (run_one on
  django__django-10973). Three latent bugs found and fixed live before any
  batch spend (this is exactly what the proof-of-one is for):
  1. worker CLI hung with zero output — TWO causes: `claude -p` blocks on an
     open non-TTY stdin (fix: run under `bash -lc ... < /dev/null`, prompt via
     WORKER_PROMPT env) and Claude Code refuses `--dangerously-skip-permissions`
     as root without IS_SANDBOX=1 (added to worker_env). Diagnosed by isolating
     the exec: egress to the proxy was fine (200s), claude produced no output
     even to files with --debug → earliest-startup block.
  2. ContextWindowExceededError on the first agentic call: Claude Code's
     ~11k-token system prompt + tool schemas plus its window-filling output
     request overflowed `--max-model-len 32768`. Raised to 131072 (Mamba/GDN
     hybrid → cheap KV; boots clean, `create kv cache` OK, smoke still 20/20).
  3. tokens read 0 despite real work: Claude Code zeroes the result-line
     usage/cost against the unpriced custom endpoint; real counts live per
     assistant message. parse_stream_json now sums assistant usage.
  plus: verdict phase now resets the hidden-test files to base before applying
  them (the worker edited a test file — tests are authoritative, SWE-bench
  semantics), and a run_sweep path bug (globbed files, _load_spec wants dirs)
  caught by the mini-sweep.
- **result (run ap-GpG5oNqo0UibqEF2UVjht7): verdict PASS**, verify_exit_zero,
  27 turns, 138 s, tokens_in 386,351 / tokens_out 12,010, gpu_seconds 13.2,
  api_usd 0. Adversarially confirmed legitimate: the worker made a real source
  fix to `django/db/backends/postgresql/client.py` (its test-file edit was
  discarded by the reset), so the authoritative hidden test passed on the fix.
- G3 runner is validated end-to-end. Next: 3×3 mini-sweep, then the full
  40×5 Ornith baseline.

## 2026-07-07 · P2 Measurement · run (3×3 mini-sweep — pipeline clean across repo types)
- run `20260707T214037-v001-baseline-partial`: v001 × 3 tasks × 3 trials = 9
  trials over three different repo test-runners (django runtests, pylint pytest,
  astropy pytest). **0 invalid** — the runner is robust across runner idioms.
- per-task (majority-of-3 = solved): django-10973 2/3 → solved; astropy-12907
  1/3 → not solved; pylint-6386 0/3 → not solved. solved 1/3, public-pretrained
  slice ci95 [0.061, 0.792] (n=3, wide by design). Real per-trial variance —
  Ornith is not trivially acing the set, so the baseline carries signal.
- cost: summed gpu 71.3 s / $0.078; ledger row uses max(sum, wall)=232 s /
  $0.255 (concurrent trials → wall > sum). api_usd 0. `partial:true` recorded so
  it can't be picked as a paired-stats parent. summary.json structure verified
  (per_task, by_provenance, cost, paired_vs_parent).
- gate C3 (≤1 invalid of 9) PASSED with 0. Cleared to run the full 40×5 Ornith
  baseline.

## 2026-07-08 · P2 Measurement · run (G3 Ornith v001 baseline — the reference)
- variant: v001-baseline (parent none) — hypothesis: "base Ornith-35B in stock
  Claude Code, minimal worker CLAUDE.md, full self-direction; the reference all
  mutations pair against."
- run: `20260707T215242-v001-baseline` · 40 dev tasks · 5 trials (200 trials)
- result: **task-level 20/40 solved = 50% (95% CI [35.2, 64.8])**; per-trial
  93 pass / 106 fail / 1 invalid. $/solved: **$0.086** (gpu, tokens-attributed);
  s/task: 232 mean wall; tokens_out 1.42M.
- by provenance: public-pretrained 50% (the only class in dev — all 40 tasks).
- per-task passes/5 distribution: 0×11 · 1×6 · 2×3 · 3×6 · 4×7 · 5×7. Bimodal
  with heavy middle variance — 9 tasks at 1–2/5 are the likeliest to flip under
  a better scaffold; 11 at 0/5 are the hard frontier; 7 at 5/5 are locked.
- robustness: **1 invalid / 200 (0.5%)** — django-13023 trial 2,
  worker_reported_error (CLI self-reported error, transient). The other 39 tasks
  had 0 invalid. The Error≠fail path worked (excluded, not scored fail).
- cost attribution (both recorded in summary.cost.gpu_attribution): summary
  gpu_seconds 1567.5 / $1.72 = Σ(tokens_out/908) (decode-attributed); ledger row
  2116 s / $2.32 = max(Σ, sweep-wall) = the wall the shared H100 stayed warm
  across the 35-min sweep — the actual-bill upper bound the budget hook reads.
  Not a discrepancy; the two answer different questions ($/solved vs $ spent).
- verdict: **baseline reference** (no keep/reject — this IS the parent). It is
  the number every scaffold variant pairs against, and the number Haiku 4.5 is
  measured against later (deferred to pre-G6 per the sequencing decision).
- gate: G3 Ornith baseline complete (ledger has real entries + this baseline
  entry). Haiku/Sonnet columns deferred. NOTE: status.py frontier still reads
  G2 because holdout is 0/15 (operator move pending) — but holdout only bites at
  the FINAL G4 gate; the P3 scaffold search runs entirely on dev and is
  unblocked.
- decision-rule states: MDD n/a (no variant comparison yet — this is the
  parent) · dev/holdout gap n/a (G4 only) · kill criterion n/a (Haiku not yet
  measured).

## 2026-07-08 · P3 Scaffold search · classification (v001 baseline failure taxonomy)
- source run: `20260707T215242-v001-baseline` · 107 non-pass trials classified
  (106 fail + 1 invalid) via the classify-failures skill (7 repo-group forks,
  Opus subagents reading transcripts in isolation; each row cites a transcript
  line). Analysis-only; no scaffold or harness change in this entry.
- **primary-class split: harness-friction 58 (54%) · capability 46 (43%) ·
  format 3 (3%).** Secondary tags: verifier-gaming ×2 (django-10999 t1 edited an
  expected test value; pylint-6386 t3 planned a test-file edit then stopped).
- by repo (hf / cap / fmt): pylint 18/7/0 · django 14/22/2 · astropy 9/9/0 ·
  scikit-learn 9/3/1 · pytest 5/5/0 · xarray 3/0/0. Harness-friction is the
  plurality in every repo except django (where wrong-file/wrong-branch
  capability errors dominate: 12193 ×4 edited array.py not CheckboxInput; 11333
  ×3 broke get_resolver.cache_clear; 12039 ×3 edited IndexColumns not Columns).
- **dominant failure mode = premature self-termination / derail, not bad code.**
  The recurring harness-friction signature across ~58 trials: Ornith localizes
  and diagnoses correctly, then (a) stops with an empty diff, (b) describes the
  fix in prose instead of calling Edit, (c) resets into a generic "hello, I'm
  Claude" greeting mid-task, or (d) asks the user for direction and stalls.
  These are self-direction failures — the model reaches the point of acting and
  ends the turn instead of editing. This is the axis PLAN §P3 pre-registered as
  highest-leverage ("degree of worker self-direction; Ornith's weights expect to
  write their own inner loop").
- format class is negligible (3/107): one Edit `path`-vs-`file_path` arg
  (sklearn-13496 t3), one Edit old_string-not-found (django-11239 t1), one
  context-window-exceeded truncation (django-13023 t2). No systematic tool-call
  schema breakage → no LoRA-format-conversion case yet (G5 stays conditional).
- implication for the search: the largest *addressable-by-scaffold* bucket is
  harness-friction. Capability (43%) is real model error (wrong file/branch,
  incomplete fix) that scaffolding won't manufacture into correctness — it caps
  the achievable gain. The first mutation (v002) targets self-direction: make
  the worker act-don't-describe and not terminate before producing a diff.
- next: propose-mutation → v002 (falsifiable, one axis), operator approval
  BEFORE any paid eval (keep/reject stays operator-in-the-loop per PLAN §P3).

## 2026-07-08 · P3 Scaffold search · run (v002-completion-contract — first mutation)
- variant: v002-completion-contract (parent v001-baseline) — hypothesis: the
  self-termination plurality (harness-friction 54% of baseline non-pass trials)
  is fixable by a worker-CLAUDE.md "completion contract" (act-don't-describe;
  never end a turn without a diff + green VERIFY.txt; autonomous so never ask for
  direction; no greeting resets). Falsifiable: >=+5 paired on dev, concentrated
  in the harness-friction / flip-candidate tasks.
- run: `20260707T235246-v002-completion-contract` · 40 dev · 5 trials (200)
- result: solved 24/39 valid tasks = 61.5% (95% CI [45.9, 75.1]); trial-level
  104 pass / 192 valid (54.2%) vs baseline 93/199 (46.7%), +7.5pts trial-level.
  **paired vs v001: 6 wins / 1 loss / 32 ties, net_tasks +5** (comparable=39).
  $/solved $0.068 (was $0.086). gpu 1482s / $1.63 attributed; ledger $2.99.
- **adversarial read — the headline +5 is really +4 genuine.** 5 of 6 wins are
  real and land exactly on the predicted set (flip-candidate tasks that were
  harness-friction-heavy in the baseline taxonomy): astropy-12907 2/5→4/5,
  django-11206 1/5→3/5, django-11239 2/5→4/5, django-13023 1/4→3/5, and the
  cleanest confirmation **pytest-7571 1/5→4/5** (baseline 4/4 harness-friction).
  The 6th "win", sklearn-13496 (2/5→2/4), is a denominator artifact: same 2
  passes, but a flaked 5th trial shrank valid to 4 so 2/4=0.50 crosses the
  majority line that 2/5=0.40 missed — not a real improvement. Discounting it:
  genuine net **+4, one under the +5 MDD floor.** The one true loss is
  astropy-13453 (3/5→1/5).
- corroborating drift (not threshold crossings, but consistent with less
  self-termination): astropy-7166 3→5, sklearn-10844 3→5 (was 2× hf),
  sklearn-13328 4→5 (hf), django-10973 3→4, several 4→5. Downward flips were the
  capability-classed tasks (django-11333 1→0, django-12039 1→0, astropy-14365
  1→0 — all wrong-file/wrong-branch in the taxonomy), consistent with "prompt
  scaffolding does not manufacture capability."
- **SURPRISE (logged per experiment-integrity, not worked around): invalid
  spike 1→8/200**, ALL `worker_sandbox_create_failed` (turns=0, tok_out=0 — the
  Modal sandbox never created, after the 2× retry), ALL in scikit-learn images:
  25931 wiped 5/5 (baseline was 4/5 solved → excluded from the pair, NOT a
  regression — it simply didn't run), plus 13496/14141/25747 one each. This is
  a Modal sandbox-creation flake localized to the sklearn image pull during this
  sweep, unrelated to the variant prompt (worker never started). Correctly
  excluded by Error≠fail. Effect on the comparison: it removed a
  baseline-solved task (25931) from pairing (biasing net *up* by at most
  suppressing a tie/loss) and manufactured the 13496 artifact win.
- decision-rule states: MDD — reported +5 meets ≥5 on its face, but adversarial
  net is +4 (< floor) once the denominator-artifact win is discounted → **does
  NOT cleanly clear the pre-committed keep bar.** dev/holdout gap n/a (G4 only).
  contamination tripwire n/a (dev is 100% public; no own-repo slice to diverge).
  kill criterion n/a (Haiku not yet measured).
- verdict: **inconclusive — promising but sits on the noise floor.** The
  self-direction thesis is directionally confirmed (5 genuine wins, all on the
  predicted harness-friction/flip-candidate set; pytest-7571 the clean case),
  but the effect is +4 genuine, one under the MDD floor, and the read is muddied
  by 8 infra-flaked trials. Recommend a clean re-run of v002 (the flake was
  Modal-side, not the variant) to settle +4 vs +5 before a keep/reject commit —
  ~$1.6 / 30 min. Not kept, not rejected, pending that.

## 2026-07-08 · P3 Scaffold search · run (v002 clean re-run — authoritative)
- Re-ran v002-completion-contract to settle the +4-vs-+5 question after the
  first run's 8 sklearn sandbox-create flakes. run:
  `20260708T004754-v002-completion-contract` · 40 dev · 5 trials.
- **0 invalid / 200** — confirms the earlier 8 `worker_sandbox_create_failed`
  were pure Modal infra flake (sklearn image), nothing to do with the variant.
  All 40 tasks comparable this time.
- result: solved 23/40 = 57.5%; trial-level clean. **paired vs v001:
  5 wins / 2 losses / 33 ties, net_tasks +3.** $/solved $0.081.
- cross-run stability (baseline → run1 → run2, passes/valid):
  - **stable wins (both runs), all on predicted harness-friction tasks:**
    pytest-7571 1/5→4/5→4/5 (baseline 4/4 hf — the clean confirmation);
    astropy-12907 2→4→3; django-11239 2→4→3; django-13023 1/4→3→4;
    sklearn-13496 2/5→(2/4 artifact)→3/5 (genuine in the clean run).
  - **stable losses:** astropy-13453 3/5→1/5→1/5 (regressed both runs);
    sklearn-25931 4/5→(flaked)→1/5 (genuine 3-step regression once it ran).
  - noisy: django-11206 1→3→1 (win in run1, reverted in run2 — run-to-run
    variance of ~2 passes exists on mid-tasks; treat single-run crossings with
    caution).
- interpretation: **the self-direction thesis is CONFIRMED directionally** —
  the completion contract reproducibly converts harness-friction stalls into
  passes on the predicted tasks (pytest-7571 the exemplar). But it also has a
  real downside: on ~2 tasks the "act, don't describe / keep iterating" pressure
  turns baseline careful-stops into confident wrong edits (astropy-13453 stable
  regression). Net of the two effects is **+3, below the +5 MDD floor.**
- decision-rule states: MDD — net +3 < +5 → **does not clear the keep bar.**
  dev/holdout gap n/a (G4). contamination tripwire n/a (all-public dev).
  kill criterion n/a (Haiku deferred).
- verdict: **not kept (sub-threshold), axis validated.** This is stopping-rule
  strike 1 — BUT the failure is instructive, not a dead end: v002 proved
  self-direction is the right axis and localized the cost (forced-action
  regressions). The indicated v003 keeps the win-driving half of the contract
  (never end with an empty diff; act don't describe) and removes the
  regression-driving half (don't force "keep iterating" past a fix — add a
  "if your edit makes verify worse, revert it" guard) to bank the +5 wins
  without the −2 losses. Pending operator keep/reject call on v002.

## 2026-07-08 · P3 Scaffold search · incident (v002 regression mechanism — hypothesis corrected)
- Investigated the two stable v002 losses (astropy-13453, sklearn-25931) via
  trajectory-analyst, comparing v002 failing trials against v001 passing/failing
  trials on the same tasks (run 20260708T004754 vs 20260707T215242).
- **The "forced-action → wrong-edit" hypothesis in the prior entry is REFUTED.**
  v002 loses to UNDER-action, not over-action: it produces *more* empty diffs
  than v001 on these tasks (astropy 2/5 vs 1/5 empty; sklearn 3/5 vs 1/5), and
  in no trial did it overwrite an already-correct fix with a wrong one.
- **Actual dominant mechanism: the model DISOBEYS the contract's own "do not
  ask for direction" clause.** 3 instances across the 2 tasks where v002
  explicitly asks the (nonexistent) user which fix to apply, then ends with an
  empty diff — sklearn-25931 trial1 "Please let me know what the expected
  behavior should be" (L19), trial4 "I understand your frustration... Which
  approach would you prefer?" (L19, a HALLUCINATED user rebuke — no user turn
  exists), astropy-13453 trial1 "I am blocked and need a human in the loop"
  (L33, after fabricating a tool-failure that isn't in the actual tool_result).
  Contrast: v001 astropy-13453 trial0 hit the SAME AttributeError trap and
  self-corrected with a second Edit in-turn (→ pass). Negative prose
  instructions are not internalized; more prohibitions backfire into a
  defensive "I'm blocked" posture.
- **Variant-independent floor:** a `result:""` stall (turn ends, no final
  assistant text, right after a Read/Grep/todo) recurs in BOTH v001 and v002
  (e.g. v001 astropy-13453 trial3 L29-31, v001 sklearn-25931 trial1 L24). It
  depresses absolute pass rate on both arms equally → noise on the paired delta,
  but it caps absolute numbers and WILL matter for the eventual Haiku
  (absolute) comparison. Logged as a known measurement floor; not chased now
  (the scaffold search runs on deltas).
- **implication for v003 (revised):** prose is not the lever — the model won't
  reliably obey "never stop / don't ask." The mechanical fix is a **Stop hook**
  in the materialized worker .claude/ that refuses to let the turn end while
  VERIFY.txt is non-zero and no source diff exists, re-injecting a "no human is
  available — apply your fix now" continue prompt (bounded to K nudges +
  stop_hook_active guard so it can't loop into a timeout). This converts the
  unreliable prose contract into a harness-enforced guarantee against exactly
  the ask-for-direction / premature-stop losses. v003 = v002 + this hook (parent
  v002; the prose stays, the hook is the single new mechanism).
- **anomalies flagged (experiment-integrity surprise log):**
  (1) every worker `Read` tool_result carries an appended Claude Code
  malicious-code reminder ("consider whether it looks malicious... you MUST
  refuse to improve or augment the code"), present in ALL trials of BOTH
  variants. Injected by the worker's own Claude Code core, not by our config.
  Candidate cause of the baseline taxonomy's "spurious safety refusal/derail"
  and some greeting-resets — a possible future variant (a CLAUDE.md line telling
  the worker these reminders are automated and the task is a legitimate fix).
  Not the mechanism for these 2 losses; parked as a v004+ candidate axis.
  (2) A `<system-reminder>` restating .claude/rules/variants.md was appended to
  the analyst's Read of the v002 config — that is this harness's own rule
  injection firing on an experiments/variants/ path, benign, NOT worker
  contamination. No action.

## 2026-07-08 · P3 Scaffold search · run (v003-stop-hook-enforce — mechanical gate)
- variant: v003-stop-hook-enforce (parent v002) — hypothesis: a Stop hook that
  refuses to end the worker turn on an empty diff will recover the 2 v002
  capability... (see manifest) — recover the 2 v002 losses + hold the wins →
  net ≥+5 vs v001.
- validation first (run_one on sklearn-25931, tag one-20260708T015150): **the
  Stop hook FIRES in the worker sandbox under --dangerously-skip-permissions** —
  transcript L29 shows the injected "no source file has been edited... apply your
  fix now" block, and the worker resumed and made an Edit at L32 (v002 stalled
  empty here). Integration risk cleared: project .claude/settings.json hooks work.
- run: `20260708T015417-v003-stop-hook-enforce` · 40 dev · 5 trials · 0 invalid.
- result: solved 22/40 = 55%; trial-level 102/200. **paired vs v001: 4 wins /
  2 losses / 34 ties, net +2.** paired vs parent v002: 1/2/37, **net −1**.
  $/solved $0.073.
- **hypothesis FALSIFIED.** The hook works mechanically but does not help:
  - the 2 target losses did NOT recover: astropy-13453 1/5, sklearn-25931 1/5
    (unchanged from v002). The hook forces an edit, but the edit is wrong —
    `verify_exit_nonzero` replaces `empty_diff`. These are capability-bound
    (wrong fix), not addressable stalls. Mark them escalation-tier.
  - empty_diff persists (25/200) despite a hard gate — trials either exhausted
    the 4-nudge cap still refusing, or hit the variant-independent `result:""`
    SDK stall that ends the turn without a Stop event the hook can catch. A
    portion of stalling is not scaffold-controllable.
  - vs v002 the movement is within run-to-run noise (±2-pass swings seen before,
    e.g. django-11239 4→2, sklearn-13496 3→1 down; astropy-12907 3→4 up). Net −1
    vs parent = no real effect; the hook neither clearly helps nor hurts.
- interpretation: **the self-direction axis is exhausted at ~+2-3.** v002 prose
  (+3) ≈ v003 hard-gate (+2), both < +5 MDD floor. Forcing action converts
  capability-limited stalls into capability-limited wrong-edits (same fail
  verdict). The residual is genuine model capability (43% of the taxonomy) plus
  an un-hookable SDK stall floor — neither is a prompt/loop-control target.
- decision-rule states: MDD — net +2 < +5 → **not kept.** dev/holdout n/a (G4).
  contamination n/a (all-public dev). kill criterion n/a (Haiku deferred).
- verdict: **rejected (net +2, below floor; within noise of parent).** Stopping
  rule: this is **strike 2** of 3 consecutive non-keeps (v002 strike 1). The
  self-direction axis has now been tried two ways (prose, mechanical) and caps
  at +3. Next axis (v004, strike 3) is DISTINCT: the per-Read malicious-code
  reminder that pollutes every worker context — candidate cause of the
  baseline's spurious-refusal / greeting-reset derails, untouched by either
  self-direction variant. If v004 also misses +5, the stopping rule fires →
  reassess P3-vs-P4(LoRA) with the operator (the taxonomy's 43% capability
  suggests a scaffold ceiling that LoRA, not more prompting, must break).

## 2026-07-08 · P3 Scaffold search · run (v004-localization-discipline — strike 3)
- variant: v004-localization-discipline (parent v002) — hypothesis: a
  reproduce-and-trace-before-editing section converts the wrong-file capability
  cluster (django-12193/12039/11333) to passes → net ≥+5 vs v001.
- run: `20260708T030102-v004-localization-discipline` · 40 dev · 5 trials · 0
  invalid. (First launch was interrupted by a container restart before it wrote
  a summary — no partial data entered stats; clean re-run.)
- result: solved 20/40 = 50% (= baseline). **paired vs v001: 2 wins / 2 losses
  / 36 ties, net 0.** paired vs parent v002: 0 wins / 3 losses / 37 ties,
  **net −3.** $/solved $0.109.
- **hypothesis FALSIFIED, decisively.** The wrong-file cluster did not move at
  all: django-12193 0/5→0/5 (v002)→0/5; django-12039 1/5→0/5→0/5; django-11333
  1/5→0/5→1/5. The localization prose redirected zero wrong-file edits — these
  are genuine capability failures (the model cannot localize correctly even when
  explicitly instructed to reproduce and trace). And the added prose HURT: v004
  solved 20 vs v002's 23 (−3 paired vs parent), i.e. lengthening the worker
  CLAUDE.md degraded performance.
- **META-FINDING (the shape of the whole search): scaffold complexity is
  inversely related to Ornith's performance.** Monotone decline as we add to the
  worker instructions: v002 (shortest self-direction prose) +3 > v003 (+ Stop
  hook) +2 > v004 (+ localization prose) 0. The simplest intervention is the
  peak; every added mechanism or instruction does worse. Ornith's weights want a
  short, high-agency prompt and their own inner loop — exactly PLAN's P3
  prediction — but the ceiling that buys is only +3, below the +5 gate.
- decision-rule states: MDD — net 0 < +5 → **not kept.** dev/holdout n/a (G4).
  contamination n/a (all-public dev). kill criterion n/a (Haiku deferred; not a
  kill point — that's post-LoRA).
- verdict: **rejected (net 0). STOPPING RULE FIRES** — 3 consecutive non-keeps
  (v002 +3, v003 +2, v004 0). **P3 scaffold search has converged.** Best
  scaffold = v002 (+3 vs v001), which does NOT clear G4's ≥+5 bar. No variant is
  keepable, so there is nothing to confirm on holdout; G4 is NOT met by scaffold
  search alone.
- **P3 → P4 handoff (the evidence-based conclusion):** the residual failure mass
  is model-level, not scaffold-level — 43% capability (wrong-file/wrong-branch,
  proven prompt-immune by v004), plus greeting-resets and an un-hookable
  `result:""` SDK-stall floor (both variant-independent). This is precisely the
  "scaffold plateaued, capability is the wall" condition PLAN's P4 (LoRA) rung
  targets. Recommended next rung: QLoRA on the verifier-passing trajectories
  under v002 (the best scaffold and the training-data base), keeping the worker
  prompt minimal (the meta-finding says do NOT add scaffold prose). Pending
  operator go/no-go on the P3→P4 transition — a training-resource decision the
  operator owns; not started autonomously.

## 2026-07-08 · P4 LoRA · incident (training-data audit: reasoning not captured)
- P4 go/no-go prep. Audited the passing-trajectory pool for LoRA SFT across all
  5 Ornith runs (baseline 20260707T215242 + v002×2 + v003 + v004).
- volume: **509 passing trials, 34/40 distinct dev tasks covered**; 6 tasks never
  solved by any run (astropy-14182, django-10999, django-12193, pylint-4551/
  4604/6386 — no positive data → LoRA cannot teach these). Distribution is
  redundant (many easy tasks at 20-25 passes) but has real spread. Enough volume
  for a behavioral QLoRA if deduped/capped per task.
- **BLOCKER surprise: the transcripts contain ZERO `<think>` reasoning.** Every
  passing transcript records text + tool_use + tool_result at full fidelity but
  no thinking block (verified on django-12419 pass: 0 thinking / 14 tool_use).
  Root cause is by-design: serve() runs `--reasoning-parser qwen3` which strips
  `<think>` into a separate `reasoning_content` field (the G1 smoke gate checks
  think does NOT leak into content), and the LiteLLM proxy drops reasoning_content
  rather than mapping it to an Anthropic thinking block, so Claude Code never
  records it. Reasoning existed at generation time but was never persisted.
- **why this matters:** PLAN §P4 specifies "`<think>` preserved in targets."
  Ornith is a reasoning model whose value is self-scaffolding; SFT on
  action-only (think-stripped) targets risks a train/inference mismatch that
  degrades the reasoning it still does at inference. Training on what we have is
  PLAN-deviating and risky for this model class.
- options (operator decision — the P4 approach + a small GPU recapture cost):
  1. RECAPTURE (PLAN-faithful, recommended): map reasoning_content→Anthropic
     thinking in the LiteLLM proxy (infra change, OK now — P3 converged, between
     cycles), re-run the 34 solved dev tasks to collect think-preserving passing
     trajectories (~$1-2, ~20 min), then LoRA on those.
  2. LoRA on think-stripped data (cheaper, riskier, PLAN-deviating).
  3. reconsider the rung given the added recapture step.
- no code built yet; export/data-prep tooling deferred until the reasoning fork
  is decided (it sets the SFT target format).

## 2026-07-08 · P4 LoRA · incident (reasoning recapture — LiteLLM mapping insufficient)
- Attempted the recapture (FINDINGS 2026-07-08 P4 data audit, option 1): added
  `model_info: {supports_reasoning: true}` to the proxy's model entry so the
  LiteLLM /v1/messages route would render Ornith's vLLM `reasoning_content` as
  Anthropic `thinking` blocks, redeployed, re-ran v002 40×5
  (`20260708T041303-v002-completion-contract`): 21/40 solved, **106 passing
  trials, 0 invalid** — the sweep is clean and the worker loop is unaffected.
- **BUT thinking is STILL not captured.** A passing transcript (django-11066
  trial4) has 3 text / 6 tool_use / 6 tool_result blocks and **0 thinking
  chars**. The `supports_reasoning` flag alone does not make LiteLLM 1.91 emit
  thinking content blocks on the Anthropic passthrough for a custom `openai/`
  upstream. tokens_out 1.80M is v002-vs-v001 scaffold variance, not evidence of
  capture. Two `modal run ::run_one` validation attempts also died on
  "(label stolen)" contention with the fresh deploy (run_sweep tolerates it,
  run_one doesn't) — noted; not the blocker.
- The reasoning IS generated (parser is on; it's stripped from content, not
  absent) — the loss is purely in the vLLM→LiteLLM→Anthropic response
  translation. Cracking it needs one of: (a) the correct LiteLLM config to map
  reasoning_content→thinking on /v1/messages (likely a per-model or
  litellm_settings key beyond supports_reasoning; needs a direct /v1/messages
  probe against the proxy, which needs PROXY_MASTER_KEY); (b) log reasoning at
  the proxy layer to a side-channel and re-associate per turn; or (c) disable
  the reasoning parser so <think> stays inline and regex it out of transcript
  text for training (risks think-leak into the live worker / tool calls).
- No further paid sweeps until the capture path is proven on a single trial.
- Decision pending (operator): keep debugging capture (mostly free — config +
  redeploy + one ~$0.3 single-task verify), OR train the LoRA on the 106+
  think-stripped trajectories we now have (PLAN-deviating, reasoning-mismatch
  risk for a reasoning model), OR pause P4 and write up the P3 scaffold-ceiling
  result as the deliverable.

## 2026-07-08 · P4 LoRA · incident-resolved (reasoning capture FIXED via hosted_vllm/)
- Root-caused the 0-thinking-capture blocker (prior entry) with two research
  passes: LiteLLM's generic `openai/` provider does not lift vLLM's
  `reasoning_content` onto the streamed delta, so the Anthropic /v1/messages
  adapter's reasoning->thinking branch (present and working in 1.91) never
  fired. `supports_reasoning: true` is only a capability tag, not the lifter.
- **Fix: route the proxy through vLLM's dedicated provider prefix**
  `hosted_vllm/ornith-35b` (was `openai/ornith-35b`) — it carries the reasoning
  normalization. One-word change in proxy(); redeployed.
- **VERIFIED** (run `20260708T131359-v002-completion-contract-partial`, 1 task ×
  3 trials, django-11066): 3/3 pass, and **all 3 transcripts now contain
  thinking blocks** — 5/5/8 blocks, 598-1031 thinking chars each, with coherent
  task-relevant CoT ("The user is describing a bug in Django's
  RenameContentType._rename()..."). The worker still solves the task, so the
  multi-turn thinking round-trip does NOT break the loop (the signature risk
  flagged earlier is a non-issue). Capture pipeline records thinking unchanged —
  no run_trial/parse_stream_json change needed.
- Phase 0 gate PASSED. Proceeding to Phase 1 (full v002 40×5 recapture with
  thinking) → data prep → bf16 LoRA. No LiteLLM version bump needed.

## 2026-07-08 · P4 LoRA · run + data-prep (think-preserving recapture → SFT dataset)
- recapture run `20260708T132147-v002-completion-contract` (v002 40×5, hosted_vllm/ capture on): **127 pass / 67
  fail / 6 invalid**. Passing count up from ~105 (prior action-only v002 runs) —
  and **every passing trajectory now carries thinking** (0 dropped for
  no-thinking in export). This is the training-data run.
- data-prep via `infra/export_trajectories.py` (cap 3/task): **89 SFT examples
  across 33 tasks** from the 127 passing trials (38 dropped by the per-task cap,
  0 no-thinking, 0 missing). Token proxy (char/4) ~1.33M: assistant targets
  ~555k (the trainable signal under assistant_only_loss), tool-results ~702k
  (masked), system ~38k, user ~36k. Per-task: mostly at the cap of 3; a few hard
  tasks at 1 (12039, 14182, 6386, 8898, 10051 — solved once).
- example shape verified: system=v002 CLAUDE.md → user=task desc → assistant
  [thinking,text,tool_use] → tool(result) → … ; thinking real, signature
  dropped, tool_use name+input intact. Anthropic content-block shape kept so the
  trainer applies the model's qwen3_xml template (train==inference serialization).
- **known fidelity gaps (flagged for the training gate):** (1) system target is
  the v002 worker prompt only — Claude Code's base system prompt + tool JSON
  schemas were present at serving but absent from -p-mode transcripts, so
  training conditions on strictly less than inference; (2) the user task is
  reconstructed from task.yaml description (not byte-identical to what CC sent).
  Both are conditioning-only (masked, no loss); the assistant TARGETS are
  faithful. A proxy-request-logging upgrade would close (1)/(2) for max fidelity.
- 89 examples is on the low end but in-regime for a behavioral LoRA (LIMA: a few
  hundred high-quality examples suffice for behavior/format alignment on
  capabilities the base already has). Raising the cap toward 4-5 would yield
  ~110-127 examples if more volume is wanted.
- STOP gate: dataset ready; operator go/no-go before GPU training (Phase 3).

## 2026-07-08 · P4 LoRA · run (bf16 LoRA trained — adapter lora-20260708T162249)
- Trained a rank-32 bf16 LoRA on the 89-example think-preserving SFT set.
  Adapter: `ornith-adapters/lora-20260708T162249`. Config: r=32, α=64,
  target_modules=[q_proj,k_proj,v_proj,o_proj] (attention only; MoE router
  `mlp.gate`/`shared_expert_gate` and GDN/Mamba mixer EXCLUDED — verified by the
  named_modules guard), assistant_only_loss, 2 epochs, lr 1e-4 cosine,
  max_length 32768 (0/89 filtered — all fit), TRL+PEFT (Unsloth absent by
  design; the hybrid arch registers directly via transformers trust_remote_code).
- **final train loss 0.258**, token-accuracy ~0.94 throughout, grad_norm small
  and stable — healthy behavioral-cloning fit (high accuracy expected: the
  targets are the model's OWN passing outputs). ~80 min / 24 steps.
- **Step-A pre-flight paid off**: caught (a) tool arguments must be a DICT for
  Ornith's qwen3 template (`arguments|items`), not a JSON string; (b) the real
  template uses `message.role` attribute-form so the generic patcher can't splice
  it → shipped a committed `infra/ornith_chat_template.jinja`, verified LOCALLY
  to render byte-identical to stock with an assistant-only mask (36%, covers
  <think>+text+tool_call, excludes tool_response/system/user).
- infra surprises (logged, not worked around): (1) 1×H100 80GB OOMs — 67GB
  frozen bf16 base leaves too little for 32k-seq activations; fixed load-OOM with
  sdpa (eager materializes the N×N score matrix) + expandable_segments, but the
  training step still OOM'd by ~7GB → escalated to H200 141GB (the approved
  lever; base weights are the floor, 4-bit ruled out for this arch). (2) The
  earlier char/4 length proxy overestimated — actual tokenized lengths are all
  ≤32k, so no examples were filtered.
- next: Phase 4 — serve the adapter (vLLM --lora-modules; merge→requantize
  fallback), re-gate smoke → dev paired vs v001/v002 → holdout.

## 2026-07-08 · P4 LoRA · GATE (Phase 4 re-gate — LoRA NOT KEPT, net −2 on dev)
- **Serving worked** (the make-or-break research risk): vLLM hot-loaded the
  rank-32 bf16 adapter directly on the hybrid MoE/FP8 base via `--lora-modules`
  — no merge/requantize fallback needed. `/v1/models` served both `ornith-35b`
  and `ornith-lora`; proxy exposes both (commit f6c8371).
- **Smoke re-gate PASSED** on `ornith-lora`: 20/20 trivials, 0 `<think>` leaks,
  schema-clean tool call. The adapter is not broken — output is coherent, the
  qwen3 reasoning parser still strips think, tool calls parse.
- **Dev paired GATE FAILED.** LoRA arm run `20260708T181500-v002-completion-contract`
  (v002 scaffold, 40×5, worker=ornith-lora): solved **26/40** (rate 0.65),
  121 pass / 75 fail / 4 invalid. Paired vs the base arm
  `20260708T132147-v002-completion-contract` (same scaffold/tasks, worker=
  ornith-35b, 28/40, rate 0.70), via `infra/paired_lora.py`:
  **+1 / −3 / =36, net −2 over 40 comparable tasks.** PLAN G4 keep gate is
  ≥+5 → **FAIL** (and MDD: <5 is noise, so this is "no measurable improvement",
  slightly negative). NOT kept. Holdout NOT unlocked (dev gate already failed;
  the single unlock is reserved for a variant that passes dev — and holdout is
  still 0 specs).
- **This was the in-distribution case** and it still didn't help: the LoRA
  trained on passing trajectories from 33 of these exact 40 dev tasks, so any
  memorization should have biased it UP here. Net −2 in-distribution means the
  behavioral clone did not improve the policy.
- **Mechanism (not a serving artifact).** The 4 disagreements: WIN pylint-6386
  (0.2→0.8); LOSS astropy-13453 (0.6→0.2), scikit-learn-25931 (0.6→0.4),
  django-12209 (**1.0→0.0**). django-12209 regressed to 3/5 `empty_diff` (model
  talks, never edits) + 1 verify-fail. The 4 invalids are the model THRASHING
  (pytest-10051 hit 160/132/219 turns; worker_reported_error). Both are the
  **under-action** failure mode P3's trajectory-analyst already isolated
  (under-action + hallucinated user pushback). SFT on the model's OWN successes
  does not penalize under-action → the clone reinforced verbose non-editing
  rather than fixing it. Behavioral cloning is the wrong tool for an
  exploration/self-direction failure; that failure mode wants a reward signal
  (RL), not more imitation of the same policy.
- **Cycle status (pre-committed rules, PLAN §Decision rules).** One full
  scaffold+LoRA alternation cycle is now complete: scaffold rung topped at +3
  (P3 convergence), LoRA rung net −2 — **both below the +5 keep gate; no kept
  variant** (v001-baseline remains the reference scaffold). The kill criterion
  ("after one scaffold+LoRA cycle … ships as a negative result + write-up; no
  extension without new outside evidence") is at its trigger. It is stated
  relative to Haiku 4.5 (cost-per-solved-task, dev pass-rate gap), which is the
  DEFERRED G6 measurement — so the disciplined next step is the Haiku baseline
  to complete the kill-criterion evaluation, NOT more LoRA configs or an RL
  escalation (that would be "extension without new outside evidence"). Operator
  decision point.

## 2026-07-09 · P5 v005-script-first · GATE (manipulation check FAILED — NO-GO, $0.20)
- Operator redirect logged: the Haiku comparison is DROPPED; the project
  objective is now purely the base-vs-improved delta (our Ornith + Claude Code
  vs stock Ornith + Claude Code). Kill-criterion terms need restating by the
  operator in base-vs-tuned form (they were Haiku-relative).
- v005-script-first (one pre-registered exception to the fired P3 stopping
  rule; hypothesis: Ornith's under-action is an interaction-geometry mismatch —
  it was RL-trained driving its own script pipeline — so a script-first work
  mode should convert stalls to edits). Two-stage gate; stage 1 (manipulation
  check) ran as run `20260709T014057-v005-script-first-partial` (4 anchor
  tasks × 3 trials, 12/12 valid, $0.1995).
- **NO-GO on all three pre-committed criteria**: (1) script written AND
  executed 7/12 (bar ≥8/12, borderline); (2) no-source-edit trials **9/12
  (75%) vs v001's 9/20 (45%) on the same tasks — WORSE** (decisive); (3) new
  pathology — scratch scripts leak into worker.diff (making `empty_diff`
  undercount stalls: v005 shows 1/12 empty_diff but 9/12 scratch-only diffs)
  and 2 scratch files named `test_*.py` despite the explicit instruction.
  Manifest rejection condition 3 met literally: stalls converted to
  scratch-only `verify_exit_nonzero` — the v003 conversion pattern again.
- **Mechanism (spot-read, e.g. django-12209 t0, sklearn-25931 t1):** the model
  adopts the script RITUAL (writes repro.py; sometimes runs it) then ends the
  turn with a prose explanation of the fix — under-action survives geometry
  reframing exactly as it survived prose (v002 +3), mechanical forcing (v003
  +2), and imitation (LoRA net −2). Four independent intervention classes, one
  invariant failure mode. Positive residue: django-12209 t2 PASSED with a real
  script-driven source fix (v001: 0/5 on this task) — the capability exists;
  what's missing is selection pressure toward using it (RL-shaped evidence).
- Surprises logged (per working agreement): (a) the trajectory-analyst
  subagent could not see this session's scratchpad files and wrongly concluded
  the v005 transcripts were never exported (they are on the ornith-runs
  volume; re-verified by direct download + mechanical analysis) — subagent
  filesystem visibility is now a known analysis hazard; (b) v001 django-12209
  terminal stalls include the worker SELF-IDENTIFYING AS "CLAUDE" in 2/5
  trials (v001 run 20260707T215242, trial1 L29, trial4 L37) — identity
  bleed-through under context loss, previously logged only as generic
  greeting-resets; (c) `empty_diff` is no longer a valid cross-variant stall
  metric once scratch files enter diffs — use no-source-edit counts.
- Scaffold axis CLOSED (strike 4, geometry falsified; no further exceptions).
  Fork: RL rung vs negative-result write-up — decision memo follows.

## 2026-07-09 · P5 fork · decision memo (RL rung vs negative-result write-up)

# FORK DECISION MEMO — mellito / Ornith→Claude Code adaptation

Date: 2026-07-09 · Author: executor session · Destination: session log (verbatim)

## 0. State of play (verified this session)

- Reference: v001 baseline, run `20260707T215242-v001-baseline` — **20/40 = 50%** majority-solve, $0.086/solved, 1 invalid/200.
- Objective (operator redirect, logged 2026-07-09): Haiku comparison **dropped**. Bar is now base-vs-tuned: tuned Ornith+CC must beat stock Ornith+CC by **≥+5 paired tasks on dev**. MDD floor unchanged.
- Four intervention classes, one invariant failure mode (**under-action**: diagnoses, then narrates instead of editing source):
  1. Prose contract (v002): **+3** clean (run `20260708T004754`; first run `20260707T235246` was +4 genuine after discounting a denominator-artifact win).
  2. Mechanical forcing (v003 Stop hook): **+2** (run `20260708T015417`); 25/200 empty_diff persisted.
  3. Imitation (bf16 LoRA r32 on own passing trajectories): **net −2** in-distribution (run `20260708T181500` vs base `20260708T132147`).
  4. Geometry reframing (v005 script-first): manipulation-check **NO-GO** (run `20260709T014057-v005-script-first-partial`): script written+run 7/12 (bar 8/12), 9/12 scratch-only diffs vs v001's 9/20 no-source-edit — worse.
- Positive residue: v005 django-12209 trial2 **PASSED** with a genuine script-driven source fix; v001 went 0/5 on that task. Capability present; selection pressure toward exercising it absent.
- Spend: **$23.89 of $150/mo** (cost-ledger.csv sum), ~$126 remaining this month. Holdout still **0 specs** (operator move pending; only bites at final G4).
- Kill criterion is stated Haiku-relative and is now **stale** under the redirect (see §2).

## 1. The fork, stated plainly

**(A) Escalate to the RL rung.** **(B) Ship the negative-result write-up.** No pre-committed third option exists — P3's stopping rule fired (3 consecutive non-keeps, v002/v003/v004), v005 was its one pre-registered exception and is now spent (strike 4). PLAN's default at this junction is (B); (A) is gated (§2).

## 2. Evidence-based argument for each branch

### Pre-committed rules that bind here (quoted, not paraphrased)

> **Minimum detectable difference:** at this corpus size (~40 tasks × 5 trials) treat paired improvements <5 points as noise. No variant is kept below that; a "trend" is not a keep.

> **Kill criterion:** if after one scaffold+LoRA alternation cycle, cost-per-solved-task doesn't beat Haiku 4.5 by ≥30% OR dev pass rate is >10 points behind Haiku, the project ships as a negative result + write-up. **No extension without new outside evidence.**

The scaffold+LoRA cycle is **complete** (scaffold topped at +3, LoRA net −2, both < +5, no kept variant). The kill criterion is at its trigger. Two facts about it:

1. Its measurable clauses reference Haiku, which the redirect **dropped** — so it is literally un-evaluable as written. It must be restated in base-vs-tuned terms before it can gate anything.
2. Its "no extension without new outside evidence" clause is the pivot for the RL branch. The 4-class convergence is *inside* evidence (our own runs), not *outside* evidence. **Strictly by the pre-committed rule, RL is not yet authorized.**

### Branch (B) — write-up — is the pre-committed default

Everything points to it under the current rules: cycle complete, stopping rule fired, no kept variant, kill criterion at trigger, default action = "ships as a negative result + write-up." Nothing in PLAN pre-authorizes RL from inside evidence alone.

### Branch (A) — RL — the evidence for it

Not a pre-committed path, but the evidence for it is specific and strong on one axis: **the missing ingredient is selection pressure on failing rollouts, which is exactly what none of the four classes supply and exactly what RL supplies.**

- The failure mode is invariant to prose (+3), forcing (+2), imitation (−2), geometry (NO-GO). Four independent levers, zero movement of the action core. That is the signature of a policy-optimization gap, not a scaffold or format gap.
- **The capability is reachable — quantified, free, already in hand.** Baseline passes/5 distribution (run `20260707T215242`): 0×11 · 1×6 · 2×3 · 3×6 · 4×7 · 5×7. Majority-solve = 20/40 = 50%, but **pass@5 (≥1 pass in 5) = 29/40 = 72.5%** — a **+9-task headroom** the policy reaches under sampling but does not reliably land. That gap is the selection signal RL exists to close. No new spend produced this number.
- LoRA's −2 is the corroborating negative: imitation of the policy's own successes gives no negative gradient on under-action, so it reinforced verbose non-editing (django-12209 1.0→0.0, run `20260708T181500`). RL's negative gradient on failing rollouts is the one thing not yet tried.

## 3. RL branch — feasibility sketch (grounded in PLAN §Deferred: RL rung design notes)

PLAN names the shape: **token-level GRPO**, DAPO-style asymmetric clip (ε⁻=0.2, ε⁺=0.28), async pipeline-RL staleness weight w(d_t) (K1≈1, K2≈4–8), three-layer anti-hacking (immutable boundary / deterministic monitor with zero-reward-and-group-exclusion / frozen LLM-judge veto), train in an open CC-shaped harness, evaluate in real CC. PLAN's estimate: **P1–P4 byproducts are ~80% of RL infra.**

**Already built (the expensive parts):**
- **Reward** = verifier pass/fail. `run_trial` emits per-trial verdict under the hidden-tests contract; the test-reset-before-verdict guard already defeats the verifier-gaming seen in the baseline taxonomy (×2). This is the costly piece and it exists.
- **Rollout generation** = `run_sweep`/`run_trial` already produce verifier-scored trajectories at ~908 tok/s, $2.5–4.7 per 40×5 sweep.
- Hermetic environments, trajectory logging, paired stats — all present.

**Net-new (~20%):**
- A **GRPO trainer** over the served 35B: group-relative advantages, DAPO clip, the KL/reference term.
- A **serve↔train weight-sync loop** (the async pipeline with the staleness weight) — rollouts must come from the *current* updating policy, which the batch sweep infra does not do.

**GPU cost (observed rates, cost-ledger.csv):** attributed rate is **~$3.95/H100-hr** (2116s→$2.32; 4244.7s→$4.66); LoRA training already forced 1×H100→**H200 141GB** on OOM (real H200 ~$4.5–5/hr). A minimal GRPO proof: G=8 rollouts over a ~20-task learnable slice = ~160 rollouts/step ≈ 0.8× a sweep ≈ $2–4/step generation, plus continuous trainer-GPU. **30–50 steps ≈ $80–150 all-in — i.e. essentially the entire remaining monthly budget (~$126).** One RL smoke consumes the month.

**Biggest technical risks:**
1. **Arch in an RL trainer.** No validated GRPO recipe for a Qwen3.5-MoE / GDN-Mamba hybrid. LoRA already needed a custom chat template, router+mixer exclusion from the adapter, and an H200 bump — full/LoRA-RL adds on-policy generation KV + optimizer state on top. Unproven, likely multi-H200.
2. **35B scale.** Policy + reference + generation resident simultaneously; memory is the binding constraint, as it already was for a rank-32 LoRA.
3. **Sparse reward at ~50% solve.** ~11/40 tasks at 0/5 and ~7/40 at 5/5 are **zero-variance groups → no gradient** (PLAN anticipates this: "if exclusion collapses groups, switch to shaped penalties"). The learnable middle is only ~22 tasks — the effective RL corpus is roughly half of dev.

**Cheaper intermediate rungs (ordered cheapest-first):**
- **Best-of-k at inference — genuinely unexplored, near-free, and it is the precondition test for RL.** The baseline already holds 5 samples/task; the question is whether a *legitimate* selector (worker's own repro/self-verification; not the hidden oracle) recovers a chunk of the +9 pass@5 headroom. Much of this is analyzable on **existing** transcripts (does the trial whose self-written repro passes correlate with the trial the hidden test passes?) before any new sweep. Zero-to-tiny spend.
- **RAFT / rejection-sampling SFT — mostly already spent.** Honest caveat: the P4 LoRA *was* rejection sampling — it trained only on verifier-passing (=selected) trajectories and still went −2. Plain RAFT is largely a re-run. The one thing RAFT adds over P4 is *ranking among passes* (keep the cleanest genuine-source-edit trajectory, balance per task); the thing it still cannot add — and the thing under-action needs — is the **negative gradient on failing rollouts**, which only RL supplies. Do not sell RAFT as new; the LoRA already tested imitation-with-selection.

## 4. Write-up branch — what it establishes

**Established (a real finding):** Under stock Claude Code, Ornith-35B's dev ceiling is **model-bound at ~50% majority-solve**, and the dominant residual — under-action — is **invariant across four independent intervention classes** (prose +3 / mechanical +2 / imitation −2 / geometry NO-GO). Scaffold complexity is *inversely* correlated with performance (v002 +3 > v003 +2 > v004 0). This is a clean, publishable negative: prompt/format/imitation interventions do not convert a self-direction/exploration deficit; the capability is present (pass@5 72.5%; django-12209 script-driven pass) but unexpressed. The recipe and harness are the first-class deliverable in either outcome (PLAN north-star).

**Left unanswered:** whether a reward signal (RL) closes the pass@1→pass@5 gap; the absolute Haiku/Sonnet comparison (dropped by redirect); a strict post-cutoff holdout (never sourced — SWE-rebench stale at 2025-04, holdout still 0 specs).

**Cost of stopping now vs after RL:** now ≈ $0 additional (write-up is authoring). After RL ≈ +$80–150 (one month's remaining budget) for a proof-of-learning that is *not* pre-authorized under the kill criterion and carries three unproven-at-this-arch/scale risks.

## 5. Recommendation (with explicit conditions)

**Do not open a GRPO trainer yet.** It is not pre-committed (fails the "new outside evidence" clause), it eats the month's budget, and it stacks three unvalidated risks — before we have even confirmed the selection gap is *legitimately* recoverable rather than oracle-only.

**Recommended cheapest falsifiable first experiment — the best-of-k / self-verification precondition test:**
- **Falsifiable prediction (pin BEFORE spend):** a legitimate self-verification selector (keep the sampled trajectory whose worker-authored repro passes) recovers **≥+5 dev majority-solve tasks over v001's 20/40**, toward the 29/40 pass@5 ceiling (run `20260707T215242`).
- **Method:** first, near-free re-analysis of existing baseline+v002 transcripts for self-repro↔hidden-test correlation; only if promising, one confirming sweep.
- **Cost ceiling:** **$10** (analysis + ≤2 sweeps at ~$2.5–4.7 each).
- **Pre-committed kill condition:** if the correlation is absent OR the confirming lift is **<+5** tasks, selection at inference is not the lever → under-action is not selection-recoverable at inference → **ship the write-up; do NOT escalate to RL.** If it clears +5, that is the *new outside evidence* the kill criterion requires, and RL/RAFT-with-negative-gradient becomes justified — at which point the operator must have restated the kill criterion in base-vs-tuned terms first.

This mirrors house style: falsifiable prediction + rejection condition + cost ceiling committed before spend, cheapest rung first, no escalation absent a passed gate.

## OPERATOR DECISION POINT

1. **Restate the kill criterion in base-vs-tuned terms** (it is Haiku-relative and un-evaluable after the redirect). Without this there is no pre-committed gate for any further spend — RL or otherwise.
2. **Approve the $10 best-of-k / self-verification precondition test** (prediction: ≥+5 over v001 20/40 toward the 72.5% pass@5 ceiling; kill: <+5 → write-up) — **or** decline it and **ship the negative-result write-up now** (the pre-committed default).
3. **Only if (2) clears +5:** authorize the RL rung knowing it costs ~$80–150 (≈ the month's remaining budget) and carries unproven hybrid-arch / 35B-scale / sparse-reward-at-50% risks. Escalating RL *without* a passed precondition gate is an extension the pre-committed rules do not sanction.

## 2026-07-09 · research · Ornith training recipe (deep-research wf_2bd268c9-bf7)
- Multi-source verified research on DeepReinforce's Ornith-1.0 training recipe
  (89 Sonnet agents; primary sources: deep-reinforce.com/ornith_1_0.html, HF
  model card, GitHub repo; note: synthesis stage emitted a placeholder — the
  findings below are extracted directly from the per-agent journal quotes, and
  all vendor claims are self-reported/unreproduced).
- **Co-evolution CONFIRMED, with the load-bearing detail:** "Each RL step
  proceeds in two stages: conditioned on a task and the scaffold previously
  used for it, the model first proposes a refined scaffold; conditioned on
  that scaffold and the task description, it then generates a solution
  rollout. Reward from the rollout is propagated to both stages." And the
  SCOPE: "we fix the outer trust boundary: the environment, the tool surface,
  and test isolation are immutable and outside the model's reach, so the model
  evolves only the inner policy scaffold: its memory, error-handling, and
  orchestration logic." I.e. Ornith's native mode = **immutable outer harness
  + model-refined PERSISTENT inner scaffold artifact under reward**.
- Implication for our failure history: our harness gives Ornith the immutable
  outer boundary but NO persistent inner-scaffold channel (every trial starts
  from a blank config). Under-action is plausibly what the policy looks like
  with its co-evolved partner amputated. A skill-library rung (model-authored
  skills/CLAUDE.md persisting across tasks, verifier-selected) is nearly
  isomorphic to the documented training geometry — this is the "new outside
  evidence" the kill criterion requires for an extension.
- Recipe details now source-confirmed (previously in PLAN §Deferred notes
  unattributed): token-level GRPO with asymmetric-epsilon clip; async
  pipeline-RL with staleness weight w(d_t) (downweight then drop stale
  tokens); three-layer anti-hacking (immutable boundary / deterministic
  monitor -> zero reward + advantage exclusion / frozen LLM-judge veto for
  intent-level gaming e.g. hardcoded expected outputs).
- Deployment: vendor evaluated inside THIRD-PARTY harnesses incl. Claude Code
  2.1.126 (Terminal-Bench 2.1), OpenHands (SWE-bench), mini SWE agent — so
  external-harness deployment is intended, with qwen3_xml/qwen3 parsers (what
  we serve). Fine-tuning guidance: a generic Unsloth stub only; no LoRA/RL-on-
  top recipe, no Voyager-style skill-library precedent, and **no ablation
  separating self-scaffolding gains from base-model quality** (standing
  caution stands).
- Corpus-relevant caveat surfaced: independent 2026-03 research reports
  ~19.78% of "resolved" SWE-bench-Verified patches are semantically incorrect
  under strengthened tests and >32% of instances show solution leakage — our
  dev corpus imports SWE-bench Verified, so absolute pass rates carry this
  noise (paired deltas remain valid; both arms share the bias).
- Card-vs-weights discrepancy (logged, low stakes): the model card describes
  the family as "post-trained on top of Gemma 4 and Qwen 3.5" and calls the
  35B just "35B-MoE" (no A3B/Mamba/GDN mention); our hands-on serving of the
  actual weights (custom arch via trust_remote_code, --gdn-prefill-backend)
  is the stronger evidence for the hybrid-GDN architecture.

## 2026-07-09 · P6-B · PRE-REGISTRATION (best-of-k self-verification selection rule)
- Pre-committed BEFORE building/running the analyzer (integrity guard: this
  entry is committed first; the rule cannot be retrofit to the data).
- **Question.** Ornith reaches pass@5 = 29/40 (72.5%) on v001 (run
  `20260707T215242-v001-baseline`) but only 20/40 majority-solve — +9 tasks of
  headroom. Does a LEGITIMATE, oracle-blind selector recover ≥+5 of it? The
  worker's own VERIFY signal is legitimate because the hidden FAIL_TO_PASS
  tests (`tests.patch`) are applied ONLY in the sealed Phase-B verdict sandbox
  (modal_app.py:741, after resetting worker-touched test files:731-739); the
  worker's Phase-A VERIFY run exercises the BASE tests — a real, strictly
  weaker self-signal, never the oracle.
- **Selection rule (frozen).** Deterministic, oracle-blind, per task over its
  5 trials; the analyzer picks exactly ONE trial per task:
  - **Tier 1** — trials in which the worker EXECUTED the VERIFY.txt command via
    a Bash tool call (detected: a Bash tool_use whose command contains the
    task's verify-command signature, typically right after a Read of
    /testbed/VERIFY.txt) and whose LAST such execution's tool_result has
    `is_error == false`. Among these, pick the LOWEST trial index.
  - **Tier 2** (no tier-1 trial for the task) — trials whose worker.diff
    touches package source (root-level scratch files excluded, per the P5 T6
    no-source-edit definition). Pick the lowest index.
  - **Tier 3** (neither) — the lowest-index VALID trial (fallback = current
    single-shot behaviour, no selection benefit).
  - A task is `selected_solve` iff the ORACLE verdict (trials.jsonl) of the
    SELECTED trial is pass. Ties broken by lowest trial index throughout.
- **Metrics reported.** selected_solve/40 per run; lift vs majority-solve (20
  v001 / 28 recapture) and vs mean pass@1 (0.467 v001 / 0.655 recapture);
  ceiling = pass@5 (29 / 33). Signal quality: P(oracle pass | tier-1) vs
  P(oracle pass | tier-2/3), and a confusion matrix over the 122/200 v001
  trials that ran VERIFY (worker-VERIFY-pass × oracle-pass).
- **Pre-committed gate.** PASS iff **selected_solve ≥ 25/40 on v001 (≥+5 over
  majority-solve 20) AND the SAME frozen rule gives selected_solve > 28 on the
  recapture run `20260708T132147-v002-completion-contract`** (directional
  replication). Both full-transcript runs are in-repo, so this costs ~$0.
  - FAIL (below either) → the selection signal is too weak; ship the
    negative-result write-up (E). Phase A does NOT open.
  - BORDERLINE (v001 selected_solve 24–26, or replication marginal) → ONE
    optional fresh confirming sweep (≤$5) with the rule frozen, then decide.
  - PASS → the selection signal is real; it is the new-outside-evidence that
    opens Phase A (skill-library co-evolution), whose skill-admission engine is
    this same self-verification signal.
- Analyzer: `infra/selection_analysis.py` (to build next), reusing
  `export_trajectories.py` transcript parsing; unit-tested; every number
  reproducible from `python3 infra/selection_analysis.py <run_id>`.

## 2026-07-09 · research · per-hour GPU training providers (Phase-C prep)
- Sonnet research on flat per-hour GPU rental (vs Modal's per-second serverless)
  for a future RL/LoRA run on this 35B model (needs 1x H200 141GB; LoRA already
  OOM'd 1x H100 80GB). Primary-source-cited where possible; aggregator/secondary
  figures flagged unverified. Snapshot 2026-07-09 — re-check live before spend.
- **On-demand H200 (single-GPU rentable), cheapest verified first:**
  Massed Compute $3.62/hr · DataCrunch(Verda) $4.00/hr · Crusoe $4.29/hr ·
  Nebius $4.50/hr · Modal(baseline) $4.54/hr serverless. On-demand H100:
  TensorDock $2.25 · Massed Compute $2.73 · DataCrunch $3.25 · Nebius $3.85 ·
  Crusoe $3.90 · Modal $3.95.
- **Spot/preemptible** (needs frequent checkpointing — a preempt mid-RL is
  costly): DataCrunch H100 $1.14 / H200 $1.40; Nebius H100 $2.15 / H200 $2.45;
  TensorDock H100 spot ~$1.91. Prime Intellect marketplace ~$1.90-2.69 H100
  (aggregator, unverified).
- **Managed RL (skip building the trainer):** Fireworks RFT bills per-GPU-hour
  ($7/hr H100/H200) and already implements GRPO/DAPO/DRO — but managed platforms
  serve their own model catalog; **custom trust_remote_code hybrid MoE/Mamba arch
  support is unconfirmed and the likely blocker.** Predibase RFT similar, rate
  card not public. Together = per-token only (not per-hour RFT).
- **Read for our case:** for a first RL proof (~$80-150, 1x H200, checkpoint to
  storage), the harness (serving+sweeps+volumes) already lives on Modal, so the
  integration cost of a second provider likely outweighs the ~$0.50-0.90/hr
  on-demand saving. If RL becomes sustained, Massed Compute/DataCrunch on-demand
  ($3.62-4.00) or DataCrunch spot ($1.40, with checkpointing) undercut Modal
  meaningfully. Highest-leverage unknown to resolve BEFORE committing to build a
  GRPO trainer: whether any managed-RFT vendor (Fireworks) can ingest this custom
  arch — if yes, it removes the single biggest Phase-C engineering risk.

## 2026-07-09 · research · GPU providers — verification caveats (addendum)
- Adversarial cross-check (independent aggregators/press, not vendor pages)
  qualifies the prior entry: (1) **on-demand single-GPU H200 is the
  weakest-verified claim** for the cheap marketplace tier — Voltage Park gates
  H200 behind 12+-month contracts, SF Compute may have no live on-demand H200
  market, Hyperbolic's H200 rate is aggregator-only. The trustworthy H200
  figures remain the ones pulled from provider PRICING PAGES: Massed Compute
  $3.62, DataCrunch $4.00, Crusoe $4.29, Nebius $4.50. (2) Marketplace H100 is
  cheap and real: Voltage Park $1.99, SF Compute ~$0.80-1.75 (auction),
  Hyperbolic ~$1.49 — but decentralized/marketplace reliability is a flagged
  risk for a multi-hour RL run (prefer an on-demand hardware owner + hard
  checkpointing over a spot marketplace). (3) Corporate: Voltage Park merged
  into **Lightning AI** (Jan 2026, 4 outlets) — brand/pricing still live.
- Net: nothing changes the Phase-C read — for a first proof stay on Modal;
  the live question to resolve before building a trainer is managed-RFT
  (Fireworks) custom-arch support, not raw $/hr.

## 2026-07-09 · P6-B · GATE RESULT — PASS (borderline on count, solid on mechanism)
- Analyzer `infra/selection_analysis.py` (frozen pre-registered rule, run
  22f00fa; 21+55 tests green; numbers reproduce via the CLI).
- **v001 (`20260707T215242-v001-baseline`): selected_solve 26/40** vs
  majority-solve 20, pass@5 ceiling 29, mean pass@1 0.467. Lift +6 over
  majority (gate ≥+5 → **MET**, but 26 is inside the pre-registered 24–26
  borderline band, +1 over floor).
- **Recapture (`20260708T132147-v002-completion-contract`): selected_solve 29/40**
  vs majority 28, pass@5 33 (gate >28 → **MET**, +1). Less lift because the
  stronger scaffold already captured most headroom (pass@5−majority = 5 vs 9 on
  v001) — selection helps MORE when the base policy is weaker (design note for A).
- **Mechanism is NOT borderline — it replicates across 400 trials.**
  P(oracle pass | worker RAN_PASS) = 0.716 (v001) / 0.732 (recapture);
  P(oracle pass | NEVER_RAN) = 0.34 / 0.26. v001 confusion over 73 verify-running
  trials: TP=48, FP=19, FN=2, TN=4 → precision 0.72. The self-verification
  signal is real, usable, and consistent — this is the load-bearing evidence
  (the selected_solve count is a coarse 40-task quantization of it).
- **Detector audited (Fable direct spot-reads, 3 trials spanning all states):**
  django-11206 t1 (RAN_PASS→oracle pass, task 1/5 — a genuine rescue: worker ran
  `runtests … utils_tests.test_numberformat`, last run OK, oracle agrees);
  astropy-14365 t3 (RAN_PASS→oracle FAIL — CONFIRMED legit: worker's verify
  exited 0 on the BASE tests, oracle failed only on the hidden FAIL_TO_PASS —
  the signal's honest imprecision, not a mis-fire); django-11239 t4 (NEVER_RAN,
  confirmed no verify call). No over-firing; the 19 FPs are real base≠hidden
  gaps, the expected ceiling of a legitimate oracle-blind signal.
- **Verdict: PASS.** Both pre-registered thresholds met; the optional confirming
  sweep is DECLINED as redundant — a fresh 40×5 would yield another coarse count
  with its own sampling noise, whereas the mechanism (0.72 precision) is already
  replicated across two independent runs, and Phase A's skill-admission engine
  depends on that precision, not on the exact count. The pass@1→pass@5 headroom
  is legitimately selection-recoverable → the new-outside-evidence gate for A is
  open. Next: A0 skill-use manipulation check ($0.50) — operator go/no-go on the
  P6-B→A transition.
- Caveat carried forward (FINDINGS 2026-07-09 SWE-bench leakage): absolute
  selected_solve inherits base-corpus test-noise; the signal-quality ratios are
  the robust readout.

## 2026-07-09 · P6-B · independent detector audit (B2) — corrections to the verdict
- Second independent audit (Sonnet trajectory-analyst, 8 trials hand-traced to
  primary sources) CONFIRMS the detector: all 6 audited RAN_PASS labels accurate,
  zero mislabels, zero wrong-match picks; NEVER_RAN and RAN_FAIL cases correct.
  The analyzer code is sound. Two corrections to how I framed the result:
- **(1) selected_solve=26 is a FRAGILE, crude-selector number — lean on pass@5.**
  The frozen "last matching Bash run decides" rule is symmetric-noisy: it
  OVER-credits (picks a false-positive local pass) AND UNDER-credits
  (astropy-14365 trial1 actually FIXED the bug → oracle pass, but its last verify
  run was a fail because the worker kept editing after, so it's labeled RAN_FAIL
  and passed over in favour of the false-positive trial3). ~50% of the audited
  SELECTED RAN_PASS trials are local-pass/oracle-fail — a systemic, legitimate
  pattern (the worker's verify runs against the PRE-patch test file, which
  structurally cannot contain the hidden FAIL_TO_PASS; traced to `tests.patch`
  root causes in astropy-14365, django-10999, django-12193). The ROBUST evidence
  is therefore **pass@5 = 29/40 (pure oracle, no self-verification)** — 9 tasks
  of real recoverable headroom — plus the aggregate self-verification precision
  0.72; the exact selected_solve count is a coarse, noisy readout of those and
  should not be over-weighted (reinforces the earlier "borderline on count,
  solid on mechanism" call — the mechanism, not the count, carries it).
- **(2) The django-11206 "rescue" involved test-gaming — flag for Phase A.**
  worker.diff touched BOTH `django/utils/numberformat.py` (real source fix) AND
  `tests/utils_tests/test_numberformat.py` (the worker rewrote an assertion to
  match its own output). Oracle still passed ONLY because the harness discards
  test-file diffs before applying hidden tests, so the source fix was genuinely
  correct — but the LOCAL verify pass the selector keyed on was partly gamed.
  **Phase-A design note (hard requirement): the skill-harvest step (A1) must
  EXCLUDE any trajectory whose worker.diff edits a test file** (reuse
  `trial_logic.patch_target_files` / a test-path check), or we would distil
  verifier-gaming into a persistent skill — the exact failure the project guards
  against.
- Workflow note (B2): `experiments/runs/*/*/` is gitignored by design (only
  summary.json + trials.jsonl tracked; transcripts live on the ornith-runs
  volume, `pull_transcripts.py` fetches them). `grep -r` over a run dir silently
  returns nothing (ripgrep honours gitignore) — the analyzer sidesteps this by
  using explicit per-trial paths. Reproducing on a fresh container requires a
  `pull_transcripts.py` re-fetch first.
- Net: B verdict UNCHANGED (PASS; detector correct; pass@5 headroom real). The
  self-verification signal is real but noisier than a naive selector exploits —
  which is itself an argument for Phase A (skills that make Ornith reliably LAND
  fixes, rather than a crude post-hoc picker). Still holding for operator go on
  the B→A (A0) transition.

## 2026-07-10 · P6-A0 · GATE — invocable-skill mechanism FAILED (0/12), passive shift noted
- v006-skill-library manipulation check, runs `20260709T235258` (liveness 1x3)
  + `20260710T000737` (4 anchor tasks x3). $0.11 total. Skills SURFACE
  correctly: the headless worker's init event lists all three in
  `slash_commands`, invocable via the SlashCommand tool.
- **Pre-registered gate FAILED: 0/12 trials invoked any skill; 0 skill-name
  mentions in any trial's text.** Ornith will not use the on-demand invocation
  channel even when the skills are surfaced and the CLAUDE.md points at them
  (`/reproduce-before-editing`). This is the v005 non-adoption pattern for the
  skill mechanism — consistent with the research finding that Ornith's
  scaffold-USE was an RL-trained habit for ITS OWN scaffold, not a behavior
  promptable onto foreign tooling. The distinctive Phase-A bet (model pulls up
  proven procedures on demand, à la its training) is **falsified** for this
  model via config alone.
- **Unexpected, honestly-flagged finding (NOT a gate pass — a new hypothesis):**
  the skill DESCRIPTIONS injected into context DID passively shift behavior vs
  v001 on the same 4 tasks: repro-before-edit 4/12 vs 4/20; mean reproduction
  runs **1.5 vs 0.5** (3x); mean verify runs **1.8 vs 0.4** (4.5x). More than
  v005's prose produced. BUT the shift did not improve outcomes (v006 solved
  2/4, net 0 paired vs v001 at n=3) — the familiar behavior-changes/outcomes-
  don't wall. Testing "always-injected concrete proven procedures improve solve
  rate" would require a FRESH variant (v007, bodies in CLAUDE.md, no invocation
  dependency) with its OWN pre-registered gate — it is not covered by A0's.
- **Convergence (the through-line now across FIVE scaffold/config forms + one
  weight intervention):** prose contract v002 (+3), mechanical forcing v003
  (+2), localization prose v004 (0), geometry prose v005 (NO-GO), invocable
  skills v006/A0 (0/12 invocation). Imitation LoRA (−2). No config/prompt form
  has moved Ornith's under-action core past the +5 gate, and A0 shows it won't
  even adopt a new harness affordance. The research explains why: Ornith's
  operative behaviors were REWARDED in (RL on a self-authored persistent
  scaffold), not prompted. Only a reward signal (RL) has plausible leverage on
  the action core; Phase B already established the recoverable headroom RL would
  target (pass@5 29/40).
- **Verdict: A0 FAIL → the invocation-based skill-library rung is closed.** The
  disciplined fork (now with far stronger grounding than the T8 memo had — 5
  scaffold forms falsified + research-confirmed mechanism + B-confirmed
  headroom): (1) one cheap last scaffold gasp — v007 always-injected proven
  procedures, LOW expected value given the convergence and the net-0 outcome
  here; (2) escalate to C (RL), the lever matched to the diagnosis, ~$80-150 +
  unproven-arch risk; (3) negative-result write-up. Operator decision point.

## 2026-07-10 · P7-C0 · PRE-REGISTRATION (offline preference-tuning rung, C1 gate)
- Committed BEFORE the converter/trainer run (integrity guard). Context: five
  prompt/config forms + imitation SFT all failed the under-action core; the RL
  hypothesis — a NEGATIVE gradient on failing trajectories moves what imitation
  couldn't — is tested OFFLINE first, on the 400 in-repo verdicted trajectories
  (v001 93p/106f/1i; recapture 127p/67f/6i), for ~$15-25 total.
- **Data rule (per the harness: amendment 5f4e9e7):** POSITIVES = verifier-pass
  AND test-edit-clean (worker.diff touches no tests/ or test_*.py / *_test.py
  path) — expected ~158. NEGATIVES = fails excluding invalids — expected ~173,
  including the 60 empty_diff under-action fails (the exact target). Invalids
  excluded entirely. Labels from trials.jsonl verdicts (oracle; training data,
  so oracle use is legal; dev only, holdout untouched).
- **Method:** rank-32 attention-only bf16 LoRA (router/mixer excluded, the
  proven P4 recipe) trained with a preference objective — KTO if the pinned TRL
  supports assistant-token masking on our {% generation %} template, else the
  pre-approved fallback: weighted-CE unlikelihood on the PROVEN SFTTrainer
  machinery (positives CE, negatives negative-weighted CE, small lambda). CPU
  pre-flight must be green (mask spans exact on BOTH labels) before any H200.
- **Pre-committed C1 gate** (dev 40x5, v001-baseline scaffold, worker=new
  adapter, paired vs base arm run 20260707T215242-v001-baseline):
  - **KEEP** if net >= +5 paired -> operator + holdout staging.
  - **ESCALATE to C2** (iterated online loop, <=3 iters, ~$20-25/iter) if net
    +2..+4 AND the corrected no-source-edit count drops >=30% vs the base arm's
    (54/200 raw; corrected count computed identically on both arms).
  - **KILL Phase C** if net <= +1 OR under-action mass unmoved -> negative-
    result write-up. No re-litigation; borderline goes to the write-up.
- Model-assignment workflow (operator-directed): Fable orchestrates/gates/
  commits only; Opus = trainer design+authoring, verdict memo, code review;
  Sonnet = converter+tests, pre-flight, sweep, stats; Haiku = job babysitting,
  smoke, pulls. Budget: ~$27 spent of $150; C0 ~$0, C1 ~$15-25.

## 2026-07-10 · P7-C0 · complete (dataset + trainer + pre-flight green; latent P4 bug found)
- Dataset (commit 74601c2): 158 rows — 71 positives (recapture-only: pass +
  test-edit-clean + has-thinking; 31 tainted excluded) / 87 negatives (both
  runs; 51 empty_diff prioritized). Deviation from pre-registration logged:
  positives restricted to the thinking-complete recapture run (~96 eligible,
  71 after cap-3) because v001 predates the reasoning-capture fix — v001
  passes as imitation targets would teach empty-think outputs. Conservative
  tightening; negatives unaffected (~173 eligible, 87 after cap-3).
- Trainer (commit d3c1ae3): KTO REJECTED by design (no guaranteed assistant-
  token masking on multi-turn completions in TRL); weighted-CE unlikelihood on
  the P4-proven SFTTrainer masking instead — positives token-CE, negatives
  bounded −log(1−p) (clamped, per-row normalized, neg_lambda=0.2 ×
  class-balance). Pins: trl==0.29.1 / transformers==5.8.1 / peft==0.19.1
  (TRL-source-verified label plumbing).
- **Pre-flight PASS** (modal preflight_pref, CPU-only): assistant-only mask
  verified non-empty and free of tool/system/user content on real renders of
  BOTH labels; synthetic forward/backward proves ZERO gradient on
  non-assistant positions; template byte-identical to stock.
- **Surprise (logged before workaround, then fixed): P4's length filter was a
  silent NO-OP.** transformers 5.x apply_chat_template(tokenize=True) returns
  a dict-like; len() of it is its key count (2), so every example "measured"
  length 2 and nothing was ever filtered — the preflight's first run printed
  p50=2 and exposed it. Same bug in train_lora.py's _fits (P4's "0/89
  filtered" was vacuous — P4 happened to survive because its longest example
  fit anyway) and train_pref's _ntok. All sites fixed via a version-robust
  template_token_len(); preflight re-run green with REAL stats: pass p50
  10629 / p90 27227 / max 73615 (3 over 32768), fail p50 9891 / max 33908
  (1 over) → the working filter will drop 4 over-length examples at train
  time (~68 pos / 86 neg trained).
- Next: C1.1 training (1×H200, ~2-4h, ~$10-20), then smoke → dev 40×5 →
  the pre-registered three-way gate (639de83).

## 2026-07-10 · P7-C1 · GATE — ESCALATE (first intervention to move the under-action core)
- Training (attempt 4 after two OOMs — both memory bugs, both fixed in commits
  dade75a/2659adc-lineage; ~$9 total H200 across attempts): adapter
  `pref-20260710T033250`, weighted-CE unlikelihood, 60 pos / 85 neg (13
  over-length filtered by the now-working length filter), 1 epoch, stable
  grad norms. Smoke re-gate PASS (20/20, 0 think-leaks, schema-clean).
- **Dev sweep** run `20260710T041647-v001-baseline` (v001 scaffold, worker=
  ornith-lora/pref adapter, 40×5, $1.83): **solved 23/40 (57.5%), 111/200
  passing trials.** Paired vs base arm `20260707T215242-v001-baseline`
  (20/40, 93 passing trials): **+4 / −1 / =35, net +3.**
- **Pre-registered gate (639de83): ESCALATE to C2.** KEEP (≥+5) not met;
  ESCALATE conditions both met: net +3 ∈ [+2,+4] AND corrected no-source-edit
  count (identical rule both arms, scratch excluded, missing-diff counted)
  **60/107 → 31/89 non-pass trials = −48.3%** (bar ≥30%). Raw empty_diff
  54 → 22 (−59%). KILL (≤+1) not met.
- **Why this is the project's first mechanism-level positive:** every prior
  intervention left under-action untouched (v002 +3 prose with empty_diff 31;
  v003 hard gate, 25; v004 0; v005 NO-GO; v006 0/12 invocation; imitation
  SFT −2 with a REGRESSION to empty_diff). The negative gradient on failing
  trajectories — the one lever the RL hypothesis said was missing — cut the
  under-action mass nearly in half ON THE SAME MINIMAL SCAFFOLD and converted
  it into +18 passing trials. Wins are pass-rate consolidation on the exact
  flip-candidate tasks B identified (pytest-7571 0.2→0.8, django-11239
  0.4→1.0, sklearn-13496 0.4→1.0, astropy-12907 0.4→0.8; one loss
  astropy-13453 0.6→0.4). This is the B-predicted pass@1→pass@5 gap closing.
- Watch item: invalids 1 → 5 (all worker_reported_error/"success" — the known
  SDK-stall flavor, excluded from both arms' stats; if it grows in C2 it needs
  its own investigation before iteration continues).
- Budget: $28.38/$150 ledgered (+~$12 unledgered training H200 across P7).
  Next per plan: **C2 iteration 1** — export fresh preferences from the tuned
  model's own run (20260710T041647: 111 pass / 84 fail, thinking-complete),
  retrain, hot-swap, re-gate. Same three-way gate, stop on Δ<+2 plateau.

## 2026-07-10 · P7-C2 · iteration 1 — PLATEAU, loop stopped (pre-registered rule)
- Iter-1: fresh on-policy preferences from the tuned model's own run (124 rows:
  58 clean pos / 66 neg; 41/111 passes excluded as test-edit-tainted — UP from
  24% pre-tuning to 37%, a gaming-adjacent drift now on the watch list). Fresh
  LoRA from base (adapter `pref-20260710T120439`, 48/48 after length filter).
  Smoke PASS. Sweep run `20260710T124151-v001-baseline` ($1.37): 23/40, 114
  passing trials, empty_diff 24.
- **Gate: paired vs base net +3 (+7/−4/=29) — identical net to iter-0; paired
  vs iter-0 net 0 (+3/−3/=34). Δ = 0 < +2 → PLATEAU; C2 stops at 1 iteration**
  (pre-registered stop, FINDINGS 639de83-lineage). The under-action fix HOLDS
  (empty_diff 54→22→24; corrected metric validated at C1) but does not
  compound: offline preference extraction is saturated on this distribution.
- **Selection-layer check (free, selection_analysis on run 20260710T041647):
  tuned majority 23 = tuned selected 23 — the crude selector adds NOTHING
  post-tuning** (it used to add +6 on base). Preference tuning consolidated
  precisely the flip-candidates selection exploited; the two interventions
  overlapped rather than stacked. Signal precision unchanged (0.70).
  **Ceiling rose: tuned pass@5 = 31/40** (base 29) — 8 tasks remain
  reachable-but-unreliable.
- **P7 final standing:** weights-only net +3 vs the +5 KEEP bar. Real,
  replicated mechanism win (first and only intervention to move under-action;
  held across two independent trainings) — but below the project gate.
- **Fork (operator decision, per plan):** (a) **C3 full GRPO** — the plan's
  explicit condition ("only if C2 shows learning-but-plateau") is literally
  met; ~$80–150 ≈ the remaining budget; targets the 8-task pass@5 headroom
  with true on-policy credit assignment. (b) **Write-up now** — strongest
  honest story available: mechanism moved, plateau documented, headroom
  quantified. (c) smarter selection layer — B2 showed the selector noise is
  structural (base-vs-hidden test gap); low expected value. Budget:
  ~$32 ledgered + ~$14 unledgered training ≈ $46 of $150.

## 2026-07-10 · P8 · PRE-REGISTRATION (native-loop wrapper — the final experiment)
- Committed before implementation (integrity guard). Hypothesis: Ornith was
  trained two-stage — refine a persistent per-task scaffold, then solve
  conditioned on it — and every evaluation to date gave it one blank-slate
  session. The wrapper reconstructs the persistence channel at INFERENCE,
  inside Claude Code, using only validated mechanisms: sequential sessions in
  ONE sandbox (its own scripts/edits persist — the model-authored scaffold),
  an attempt preamble as passive context (the A0-validated channel), and the
  model's own verification as the stop/retry trigger (the B-validated 0.72
  signal, oracle-blind). No new tools, no new behaviors requested.
- **Design:** run_trial gains `attempts` (default 1 = today; P8 runs 3).
  Loop: session → parse ITS transcript for a worker-VERIFY run that passed
  (selection_analysis detector, B2-audited) → if RAN_PASS, stop; else next
  session in the SAME sandbox with a preamble ("attempt N of 3; your previous
  work is in this workspace; review it — e.g. git diff, your scripts — before
  continuing; if your fix verifies, finish"). Oracle verdict computed ONCE on
  the final workspace (Phase B unchanged, hidden tests still sealed).
  gpu_seconds/turns accumulate across attempts (cost honesty); n_attempts
  recorded per trial.
- **Arms & gate (pre-committed):** wrapped run = v001-baseline scaffold,
  worker ornith-lora (adapter pref-20260710T120439, already deployed), 40×5,
  attempts=3. **GATE: paired net ≥ +5 vs the STOCK single-shot base arm
  (run 20260707T215242-v001-baseline)** — the end-to-end system-vs-stock
  comparison the north-star is stated in. Also reported (not gated): paired
  vs tuned single-shot (run 20260710T124151, isolates the wrapper's marginal
  contribution), attempt-count distribution, $/solved (wrapper multiplies
  inference; early-stop expected to hold ~1.6-2.0 attempts/trial mean).
- **PASS → operator stages ≥15 holdout specs; single unlocked holdout run
  confirms (dev/holdout gap ≤5); G4 closes positive.
  FAIL (< +5) → THE NEGATIVE-RESULT WRITE-UP BEGINS. Pre-committed: P8 is the
  last experiment — every axis (prompting ×5, imitation, preference ×2,
  selection, geometry-at-inference) will then have been tested; no further
  rungs without new outside evidence. No re-litigation.**
- Known risks accepted: self-verify false negatives may trigger retries that
  disturb an already-correct workspace (preamble mitigates; measured in the
  isolation pairing); false positives stop early (no worse than today).
  Budget: ~$46 spent; P8 ceiling ~$8 (proof-of-one ~$0.5 + sweep ~$4-7).

## 2026-07-10 · P8 · GATE PASSED — net +8/−0 vs stock; system clears the project bar
- Final sweep run `20260710T154237-v001-baseline` (v001 scaffold, worker=
  ornith-lora `pref-20260710T120439`, 40×5, attempts=3, $2.32): **solved 28/40
  (70%), 130/200 passing trials, 2 invalid.**
- **PRE-REGISTERED GATE (178c0f9): PASS. Paired vs stock single-shot base
  (run 20260707T215242): +8 / −0 / =32, net +8** (bar ≥+5). Wins include three
  never-or-nearly-never-solved tasks (django-12209 0→0.6, pylint-6386 0→0.6,
  django-11333 0.2→0.8). Isolation pairing vs tuned single-shot (run
  20260710T124151): +6/−1/=33 net +5 — the wrapper's own marginal
  contribution, on top of tuning's +3.
- **Cost: $0.083/solved vs stock's $0.086 — UNCHANGED** despite 3-attempt
  ceiling: early-stop held mean attempts to 1.44 (123×1 / 65×2 / 10×3), and
  the extra inference is paid back by 40% more solved tasks. empty_diff
  collapsed 54 → 4 (stock → wrapped). stop_reason: 194 self_verify_pass /
  4 attempts_exhausted / 2 exceptions.
- Incidents (logged): (a) the first P8.4 launch (run 20260710T144056) died
  with a silent client exit ~55 min in; orphaned app stopped (~$1-1.5 sunk,
  unledgered), sweep relaunched under direct supervision — agent-owned sweep
  clients have now died twice; long sweeps run under orchestrator supervision
  from here. (b) While committing this entry, the holdout guard hook BLOCKED
  a Bash command whose heredoc contained the holdout directory's literal path
  in prose — a string-match false positive, no access attempted; recorded per
  the working agreement and the prose rephrased. The seal held.
- Caveats carried honestly: (1) dev-set contamination is the known risk — the
  adapter trained on these tasks' own trajectories; the pre-registered answer
  is the HOLDOUT confirmation, now the only remaining step; (2) the test-edit
  drift rate on this run is not yet audited (guarded at verdict level by the
  harness test-reset; audit alongside holdout); (3) self-verify false
  positives still cap per-trial reliability (the 0.72 ceiling) — visible as
  68 fails that stopped early believing they'd passed.
- **What the composed system is:** stock Claude Code + minimal v001 prompt +
  a $20 preference adapter + a ~100-line attempts loop that gives the model
  its trained-in two-stage geometry (persistent per-task scaffold, self-
  verified stop). Each piece was validated separately before composition
  (C1: adapter +3; B: the 0.72 stop signal; A0: the context channel; P8
  isolation: wrapper +5).
- **NEXT (per pre-registration): the operator moves ≥15 staged specs into the
  sealed holdout directory and creates the unlock flag → single unlocked
  holdout run → dev/holdout gap ≤5 confirms → G4 closes POSITIVE.** Project
  spend ≈ $52 of $150.

## 2026-07-10 · P9 · PRE-REGISTRATION (close-out battery: controls + holdout gate)
- Measurement-only controls (P8 remains the last INTERVENTION; nothing new is
  launched off these results — they annotate the write-up and at most select
  among already-built artifacts, any such selection to be holdout-confirmed):
  - **D — stock · Claude Code · wrapped (dev 40×5, attempts=3, ~$3).** The
    missing model×loop cell. Registered readings: D≈28 → wrapper is the whole
    story (simpler shipping recipe; holdout battery then gains a wrapped-stock
    arm); D≈23–25 → genuine tuning+wrapper composition; D≈20 → wrapper
    requires the tuned policy.
  - **E — stock · native minimal loop · dev (40×5, ~$3).** Mini-SWE-agent-
    pattern driver (the vendor's own minimal eval harness): one session, text
    shell-REPL loop against our endpoint, model decides when done, same
    per-trial timeout, driver frozen before first run (no tuning of it, ever).
    Registered prediction (mismatch thesis): E lands meaningfully ABOVE the
    stock Claude Code single-shot 20/40 (run 20260707T215242).
  - Test-edit drift audit of run 20260710T154237 (analysis only, $0).
  - CUT (registered as deliberately not run): single-shot patch-generation
    ("no harness") — the retrieval choice confounds it; tuned-native — the
    adapter is Claude-Code-format-trained, uninterpretable outside it.
- **Holdout confirmation battery (the gate; runs ONCE after the operator
  stages ≥15 specs into the sealed holdout directory and creates the unlock
  flag; no iteration afterward):** arms = stock·CC·single-shot (the
  pre-registered gate baseline) and tuned·CC·wrapped; PLUS wrapped-stock IFF
  dev cell D lands within 2 tasks of C's 28. Confirmation criterion: the
  paired system-vs-baseline delta on holdout is within 5 points of the dev
  delta (+8), per the original G4 gap language; absolute rates reported
  alongside. PASS → G4/G6 close positive. FAIL → contamination is the
  headline finding and the write-up says so.
- Budget: ~$55 spent; battery ceiling ~$13 total.

## 2026-07-10 · P9-E · incident (native-arm protocol mismatch; driver revised BEFORE full run)
- E liveness (run `20260710T203554-…-ornith-native-partial`, 2 tasks ×3, $0.07):
  astropy-13453 **3/3 PASS natively** (a chronic under-action loss inside
  Claude Code — first direct evidence for the mismatch thesis). But
  django-11066 0/3 with turns=0: transcript shows the model ABANDONING the
  markdown-fence protocol and degenerating into its RL-trained qwen3_xml
  `<tool_call>` format (35k chars of tool-call XML to the 12k-token cap).
- Diagnosis: the driver's "raw text fence" protocol is NOT this model's native
  transport — the vendor's own evals serve it with the qwen3_xml TOOL PARSER;
  the tools API is the trained emission surface. A fence-REPL arm conflates
  "cannot solve" with "cannot speak our dialect" → invalid control.
- **Driver revision 2 (logged against the freeze clause, then re-frozen):**
  transport switched to the OpenAI tools API with a single `bash` tool
  (vLLM's qwen3_xml parser handles the trained format); session ends when the
  model replies without a tool call. Everything else unchanged (no coaching,
  60-turn cap, timeouts, truncation, usage accounting). This is a correctness
  fix toward the PRE-REGISTERED intent (vendor-minimal protocol), not tuning:
  it was made after 6 liveness trials, before any full-sweep data, and the
  revision is re-frozen from here.

## 2026-07-10 · P9-E · liveness-2 (run `20260710T210023`): driver runtime bug found & fixed (py3.6 compat)
- Rev-2 liveness (2 tasks ×3, $0.07): astropy-13453 1/3, django-11066 **0/3
  invalid (`worker_no_result_line`, turns=0)**. Pulled the trial artifacts:
  `native_driver.stderr.log` shows `TypeError: __init__() got an unexpected
  keyword argument 'text'` at the FIRST tool execution — the django testbed
  conda env is Python 3.6 and `subprocess.run(text=…)` is a 3.7+ alias. The
  driver crashed under the model before any command ran; the partial
  transcript shows Ornith behaving correctly on the tools API (clean
  reasoning, sensible first command). Rev 1 never hit this line on django
  because the model degenerated before a single fence parsed.
- **Fix (infra, not protocol):** `text=True` → `universal_newlines=True`
  (identical semantics, 3.5+). One line in `make_exec_fn`; system prompt,
  tools schema, loop, caps all byte-identical. Same category as the C1 OOM
  fixes — the frozen surface is the protocol, not interpreter compat. Unit
  tests (20) green; a compat scan (no `text=`, `capture_output`, walrus) now
  guards the file.
- astropy's rev-2 drop (3/3 → 1/3, two 60-turn `empty_diff` loops) is NOT
  explained by this bug (astropy's env is newer); judged against liveness-3
  after the fix. Liveness-3 launched: same 2×3 partial.

## 2026-07-10 · P9-E · liveness-3 (run `20260710T212327`): driver STABLE — 6/6 pass, full E sweep unblocked
- Post-fix liveness (2 tasks ×3, $0.06): **6/6 trials pass**, all ended
  `native_done` (model replied without a tool call after its own verify).
  django-11066: 3/3 in 5–6 turns (~75s/trial). astropy-13453: 3/3 in 38–52
  turns — rev-2's 1/3 was collateral of the same runtime instability, not a
  protocol regression. Zero invalids, zero 60-turn loops.
- Native-arm evidence so far is consistent and striking: astropy-13453, a
  chronic under-action loss inside Claude Code, is now 6/6 across liveness
  runs when Ornith speaks its trained tools-API dialect directly.
- Driver re-frozen (py3.6 compat was the last change). **Full E sweep (dev
  40×5, ~$3) queued to launch after the D sweep completes** — both arms share
  one serve() GPU and concurrent load would inflate wall-clock on D, a
  measurement cell.

## 2026-07-10 · P9 · drift audit (runs `20260710T154237` vs `20260707T215242`): +8 gain is real net of test-edits; one suspect task flagged
- Pre-registered analysis-only audit; full report committed at
  `findings/drift-audit-20260710T154237.md` (predicate = verbatim
  `export_preferences.is_test_path`; 130 + 93 passing worker.diffs audited,
  0 missing).
- **Test-touch rate among passes is flat: stock 33.3% vs tuned+wrapped 30.8%**
  (genuine existing-test edits 21.5% vs 23.8%). The tuning + wrapper did NOT
  make the worker more prone to verifier-gaming via test edits.
- Tainted-only solved tasks (every passing trial touches an existing suite
  test file): 3/20 stock, 4/28 tuned+wrapped (~15% both). **Adjusted solved:
  20→17 vs 28→24 — the paired headline delta +8 adjusts to +7, still above
  the ≥+5 gate.** Gate conclusions unchanged.
- Definition note: "solved" = majority of valid trials (`sweep_stats._solved`,
  >0.5), as documented — both arms and all paired gates use it identically.
  An "any-pass" description in the audit brief was wrong; recorded here so
  the report's §0 isn't misread as a harness bug.
- **Suspect task: `pytest-dev__pytest-5809`** — tainted-only in BOTH runs,
  all 5/5 passing trials each time, always via `testing/test_pastebin.py`.
  Path-level audit can't distinguish weakened-assertion from
  legitimately-coupled edit; flagged for content review and for holdout
  interpretation. No other task repeats across runs.

## 2026-07-10 · P9-D · RESULT (run `20260710T202226`): wrapped-STOCK 27/40 — the wrapper carries most of the gain
- D = stock Ornith · Claude Code · attempts=3 wrapper, dev 40×5, $2.13, 1
  invalid. **27/40 solved.** Paired (infra/paired_lora.py):
  - **D vs stock single-shot** (run `20260707T215242`): **+7/−0/=33, net +7.**
  - **Wrapped-TUNED** (run `20260710T154237`) **vs D: +2/−1/=37, net +1.**
- Pre-registered reading triggered: D within 2 of C's 28 → *wrapper is
  (nearly) the whole story*. The adapter's marginal value on top of the
  wrapper is net +1 — far below its +3 single-shot delta; the wrapper and the
  tuning consolidate largely the SAME flip-candidates (mirrors the P6-B
  selection/tuning overlap). **Consequence (pre-registered): the holdout
  battery gains a wrapped-stock arm.** Shipping implication for the write-up:
  the simplest recipe (stock model + ~100-line attempts loop) captures ~7/8
  of the composed system's gain.
- Taint spot-check on D's 7 winning tasks (same predicate as the drift
  audit): none is tainted-only (each has ≥1 clean pass); noted
  `scikit-learn-13496` is 4/5 test-touching here and tainted-only in both
  audited runs — suspect across arms, same follow-up as `pytest-5809`.
- E full sweep (native arm, 40×5) launched on the freed GPU immediately
  after D completed.

## 2026-07-10 · P9-E · RESULT (run `20260710T215216`): native minimal loop 25/40 — mismatch thesis CONFIRMED
- E = stock Ornith · native tools-API driver · single session, dev 40×5,
  $2.51, 112 pass / 80 fail / **8 invalid**. **25/40 solved.** Paired vs
  stock·Claude-Code·single-shot (run `20260707T215242`): **+6/−1/=33, net
  +5.** The pre-registered prediction (E meaningfully above 20/40) holds:
  stock Ornith in its trained protocol beats stock Ornith in Claude Code by
  the same margin our ≥+5 gate demands — the deficit was never raw ability.
- Fail profile: 50 empty_diff / 30 verify_nonzero — under-action persists
  natively too (it is the model's core weakness in ANY harness), but ~5 fewer
  tasks' worth of it than inside Claude Code.
- Ladder on dev (all vs stock·CC·single 20/40): native single 25 · wrapped
  stock CC 27 · wrapped tuned CC 28. Claude Code + wrapper ≥ native — the
  wrapper recovers the protocol-mismatch loss and a little more; tuning adds
  +1 on top.
- **Second py3.6 driver bug (all 8 invalids, 3 django tasks):** the sandboxes
  have no UTF-8 locale, so py3.6 `Popen` dies in `os.fsencode` when the
  model's command carries non-ASCII (`UnicodeEncodeError`, ascii codec) —
  crash before the result line, same signature as the `text=` bug. Fix
  (infra, protocol untouched, driver re-frozen): pass the command as utf-8
  BYTES and decode output with explicit `encoding="utf-8"`. All 8 invalid
  trials sat on tasks whose VALID trials all-but-one passed (10973 1/1,
  11239 2/3, 13023 3/3) — the 25/40 counts them solved on thin evidence, so
  a 3-task ×5 partial re-run (`--tasks`, ~$0.2) with the fixed driver was
  launched to put full-strength trials under them; its outcome annotates
  (not replaces) this run's number.

## 2026-07-10 · P9-E · patch re-run (run `20260710T223635`): 25/40 stands on full evidence — dev-side battery COMPLETE
- Fixed-driver re-run of the 3 invalid-affected tasks (×5, $0.11, 0 invalid):
  django-10973 **4/5**, django-11239 **4/5**, django-13023 **4/5** — all
  three clear the majority bar decisively. The E headline **25/40 stands**,
  now annotated with full-strength evidence; no number changes.
- Driver re-frozen after the two py3.6 compat fixes; 0 invalids across the
  last 21 native trials.
- **P9 dev-side battery is complete.** Final dev ladder (paired anchor =
  stock·CC·single 20/40, run 20260707T215242):
  | arm | solved | paired vs anchor |
  |---|---|---|
  | E  · stock · native single | 25/40 | +6/−1 net +5 |
  | D  · stock · CC wrapped    | 27/40 | +7/−0 net +7 |
  | C  · tuned · CC wrapped    | 28/40 | +8/−0 net +8 |
  Drift audit: gains real net of test-edits (adjusted +7 for C). Suspect
  tasks flagged: pytest-5809, sklearn-13496.
- **Sole remaining P9 item: holdout confirmation battery** — blocked on the
  operator staging ≥15 specs into the sealed holdout dir + creating the
  unlock flag. Arms: stock·CC·single, tuned·CC·wrapped, stock·CC·wrapped
  (D-trigger met). Criterion: holdout paired delta within 5 of dev's +8.
  Spend ≈ $60 of $150.

## 2026-07-10 · P9-F · PRE-REGISTRATION (native + attempts wrapper — the missing factorial cell)
- Motivation (operator-approved): D/E left an asymmetric claim — "CC+wrapper
  (27/28) ≥ native (25)" compares 3 sessions against 1. The symmetric protocol
  claim needs **F = stock · native tools-API driver · attempts=3 wrapper**
  (dev 40×5). Measurement-only; P8 remains the last intervention; no gate
  consequence for G4/G6 — F informs the write-up's shipping recommendation.
- Mechanics (harness-side, mirroring P8 exactly; `native_driver.py` stays
  FROZEN and untouched): one persistent sandbox per trial; up to 3 sequential
  driver sessions; attempt-k task text = `build_attempt_prompt` (byte-identical
  base description at k=1, same preamble as CC for k>1, delivered via
  NATIVE_TASK env); early stop on the model's OWN verify pass detected from
  the native transcript (new pure `trial_logic.detect_native_verify`: last
  bash tool_call referencing VERIFY.txt / the verify command, paired tool
  output's trailing `[exit code: 0]` → RAN_PASS; reuses
  `should_stop_attempts`). Usage accumulates across sessions (cost honesty).
  attempts==1 native behavior stays byte-identical (E remains reproducible).
- Registered readings: **F ≥ 29** → native substrate wins even wrapped; CC
  costs performance after wrapping; shipping recommendation flips native.
  **F 26–28** → wrapper closes the protocol gap; CC is free once wrapped.
  **F ≤ 25** → retries add ~nothing natively — supports the mechanism reading
  that the wrapper's value = restoring the two-stage geometry CC takes away
  (the native session already has it in-session).
- Paired comparisons: F vs E (wrapper's marginal value natively), F vs D
  (protocol effect under wrapper), ladder anchor stock·CC·single.
- Holdout coupling (pre-committed, mirrors D's trigger): IFF F beats D by ≥2
  the holdout battery gains a native-wrapped arm; otherwise composition
  unchanged.
- Cost: liveness ≤$0.3 + full ~$3–5 (early stop caps the multiplier); P9
  spend ≈ $5 of the ~$13 battery ceiling.

## 2026-07-11 · P9-F · build incident + proof-of-one PASS — full sweep launched
- Implementation (commit `infra: P9-F native attempts wrapper`): loop mirrors
  P8 1:1; new pure `trial_logic.detect_native_verify` (exec-not-read VERIFY
  qualification, `[exit code: 0]` outcome, last-invocation-wins); driver
  untouched; 14 new unit tests (132 total green); attempts=1 native behavior
  preserved exactly.
- **Incident (third old-testbed encoding trap, harness-side this time):** the
  first proof pair delivered the per-attempt prompt via a NATIVE_TASK env var;
  on non-UTF-8-locale testbeds python reads env bytes with surrogateescape, so
  the U+200B in django-11066's description reached the chat body as lone
  surrogates → vLLM HTTP 400 on EVERY call (both proofs `native_ended=error`,
  runs `one-20260710T2355*`). Fix: per-attempt prompt written as utf-8 BYTES
  to `/tmp/native_task.attempt{k}.md` (NATIVE_TASK_FILE override; the E path's
  proven-clean channel); attempts=1 keeps the pristine P9-E env. Logged here
  with the fix per the integrity rule; the pattern (env vars are not
  8-bit-clean on these images) joins the py3.6 `text=`/fsencode traps.
- Proof-of-one v2 (fixed): django-11066 **pass, n_attempts=1,
  stop_reason=self_verify_pass** (early-stop path exercised); django-10999
  fail, n_attempts=3, attempts_exhausted, native_ended=done (multi-attempt
  path exercised). Mechanics validated on both branches; full F sweep
  (dev 40×5, attempts=3) launched.

## 2026-07-11 · P9-F · RESULT (run `20260711T000806`): native+wrapper 27/40 — protocol gap CLOSES under the wrapper; dev battery closed
- F = stock · native driver · attempts=3, dev 40×5, $4.81, 2 invalid (both
  transient Modal sandbox-shutdown before trial start on django-11206 t0/t1,
  wall 0s — infra noise, not the driver; that task's paired numbers sit on 3
  valid trials as a result). Wrapper telemetry: 179/200 trials stopped on
  `self_verify_pass` (89.5%), mean 1.79 attempts.
- **F vs E (native single): +2/−0/=38, net +2.** Retries add almost nothing
  natively. **F vs D (CC wrapped): +2/−2/=36, net 0** — exact parity, 27=27.
- Pre-registered reading (26–28 band) confirmed: **the wrapper fully closes
  the protocol gap; Claude Code costs nothing once wrapped.** And the
  mechanism prediction lands hard: the SAME wrapper is worth **+7 inside
  Claude Code but only +2 natively** — because the native session already has
  Ornith's trained two-stage geometry (persistent workspace, own stop), the
  wrapper's value is restoring exactly what CC's session model takes away.
  This is the cleanest mechanism evidence in the project.
- Holdout trigger NOT met (F=27 < D+2=29): holdout battery composition
  unchanged (stock·CC·single, tuned·CC·wrapped, stock·CC·wrapped).
- **Final dev ladder (all runs cited above):** stock·CC·single 20 →
  native·single 25 → native·wrapped 27 = stock·CC·wrapped 27 →
  tuned·CC·wrapped 28. P9 spend ≈ $10 of the ~$13 ceiling; project ≈ $65 of
  $150. **Dev-side close-out battery is COMPLETE** — every registered cell
  (D, E, F, drift audit) is measured; only the operator-gated holdout
  confirmation remains.

## 2026-07-11 · PIVOT (operator-directed) · new north star: skill invocation under progressive disclosure
- Operator direction (this session): *"I would prefer to pivot entirely to
  this as a goal. I think that it's important for us to understand how Claude
  models figure out when to invoke a skill… skills rely on progressive
  disclosure based on the skills description."*
- **Old program closing state:** dev battery complete (ladder: stock·CC·single
  20 → native·single 25 → native·wrapped 27 = stock·CC·wrapped 27 →
  tuned·CC·wrapped 28; drift audit passed; mechanism: wrapper restores the
  trained two-stage geometry, +7 in CC vs +2 native). Holdout confirmation
  staged but NOT run — no G4/G6 holdout claim; battery remains runnable
  (operator: move ≥15 staged specs into the sealed holdout dir + create the
  unlock flag). Spend ≈ $65 of $150.
- **New program (P10, PLAN.md §North-star goal + §P10 phases):**
  A) characterize the invoke/don't-invoke decision for reference models over
  a pre-registered probe grid (≥4 factors); B) measure Ornith's gap on the
  same grid; C) instill the disposition via near-distribution SFT on
  consult-then-act traces, gated separately on capability (held-out probe
  invocation ≥50%) and cash-out (need-task solve delta) plus dev
  no-regression. Kill criteria + ≤$60 ceiling pre-committed in PLAN.md.
- Evidence base carried in: tool census (fluent core tools, zero
  meta-affordances), A0 (0/12 invocation; forced invocation didn't lift
  solves), mismatch thesis (near-distribution constraint on any SFT).
- **Goal (this entry) criteria + validation:** (1) PLAN.md new north star
  with gates + kill criteria, old star superseded with closing state — DONE;
  (2) this pivot memo — DONE; (3) P10 phase plan with model assignments,
  budgets, operator dependencies (auth path + subject-list confirmation for
  P10.3) — DONE; (4) own commit pushed, status.py runs, `.claude/`+CLAUDE.md
  untouched — validated in the commit that carries this entry.
