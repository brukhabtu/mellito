# ornith-harness

Adapting Ornith-1.0-35B into a Claude Code worker via scaffold search + LoRA.
Start here: PLAN.md (goals, decision rules, **§Status**) → CLAUDE.md (working
agreements) → findings/FINDINGS.md (project memory). Gate state:
`python3 infra/status.py`.

Status in one line: G1 serving MET (compiled vLLM, ~908 tok/s); G2 dev MET (40
eval-ready tasks), holdout 18 staged; G3 runner validated (proof-of-one PASS,
clean mini-sweep) with the Ornith v001 baseline running. Next: the P3 scaffold
search delta loop; hosted-Claude baselines deferred to pre-G6. Details: PLAN.md
§Status.

Fresh-container setup: run the `/modal-auth` skill (headless Modal login), then
`modal deploy infra/modal_app.py` to start the endpoint. Corpus tooling:
`infra/import_swebench.py` (dev), `infra/import_swebench_live.py` (holdout
staging), `infra/determinism_check.py` (3+3 admission).

CI: the CI workflow template `infra/ci/classify.yml` (copy to `.github/workflows/`) runs the classify-failures skill on an
eval run via the Claude Code GitHub app (read-only; Modal creds come from repo
secrets, which reach Actions runners). Needs the app installed +
`CLAUDE_CODE_OAUTH_TOKEN` (subscription) and `MODAL_TOKEN_ID/SECRET` secrets.
Pull a run's transcripts locally with `infra/pull_transcripts.py <run_id>`.
