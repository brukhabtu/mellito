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
