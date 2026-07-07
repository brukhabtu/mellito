# ornith-harness

Adapting Ornith-1.0-35B into a Claude Code worker via scaffold search + LoRA.
Start here: PLAN.md (goals, decision rules, **§Status**) → CLAUDE.md (working
agreements) → findings/FINDINGS.md (project memory). Gate state:
`python3 infra/status.py`.

Status in one line (2026-07-07): G1 serving MET (compiled vLLM, ~908 tok/s
aggregate); G2 dev MET (40 eval-ready tasks), holdout 18 staged for operator
move; G3 measurement machinery built, first baseline pending. Details: PLAN.md
§Status.

Fresh-container setup: run the `/modal-auth` skill (headless Modal login), then
`modal deploy infra/modal_app.py` to start the endpoint. Corpus tooling:
`infra/import_swebench.py` (dev), `infra/import_swebench_live.py` (holdout
staging), `infra/determinism_check.py` (3+3 admission).
