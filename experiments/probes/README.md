# experiments/probes/

Generated probe workspaces for the P10 skill-invocation grid live here at run
time under `experiments/probes/<run_id>/<cell>/<trial>/` — each holding a
`.claude/skills/<name>/SKILL.md` tree (the SUBJECT under test, never this
repo's own `.claude/`), the task `prompt.txt`, a `cell.json` provenance record,
and, after a run, a `transcript.jsonl`. They are DATA, materialized by
`infra/probe_harness.py:materialize_probe(...)`, exactly like
`experiments/variants/*/claude-config/`. Nothing under this directory is
committed except this README; generated workspaces are disposable and
reproducible from the cell registry + trial index (string-seeded RNG).
