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
