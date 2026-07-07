"""Pure trial-orchestration logic for run_trial — no Modal imports, so the
stage→verdict table, the Claude-Code stream-json parser, the worker env, and
the image build commands are all unit-testable without a GPU or a container.

run_trial (in modal_app.py) is the only caller; it owns the sandboxes and I/O.
This module owns the *decisions*: what a stage means, what the tokens were, what
environment the worker runs under. Keeping them here means a verdict rule change
is a table edit with a test, never a container round-trip.
"""

import json

# Pinned so the worker image is reproducible run-to-run (a version drift would
# silently change the scaffold under test). Bump only in a `harness:` commit.
CLAUDE_CODE_VERSION = "2.0.14"

# Aggregate decode throughput of the shared vLLM fleet (tok/s), measured on the
# one H100 under concurrent batching. GPU-seconds attributed per trial =
# tokens_out / this rate; it is a *cost attribution* constant, not a claim about
# any single request's latency. The ledger cross-checks against wall time.
AGG_TOK_PER_S = 908.0


def parse_stream_json(lines: list[str]) -> dict:
    """Extract usage/verdict signal from a Claude Code `stream-json` transcript.

    The CLI emits one JSON object per line and terminates with a
    `{"type":"result", ...}` summary carrying num_turns / cost / is_error. We
    scan for the LAST valid result line (a run can legitimately contain several
    if it was resumed); everything else — partial lines, banner text — is
    tolerated and skipped.

    TOKENS come from the per-`assistant`-message `usage`, NOT the result line:
    against a custom endpoint (ornith via the proxy) Claude Code leaves the
    result-line `usage`/`total_cost_usd` all-zero (its cost DB has no
    'ornith-35b' entry), but every assistant message still reports its own
    token usage. Summing those is the only source that is correct for BOTH the
    custom endpoint and the real Anthropic API. `tokens_in` folds all three
    input buckets (fresh + cache read + cache creation) and therefore counts
    the per-turn re-sent context — i.e. total prefill work, not unique tokens.
    tokens_out is the decode work that drives GPU-seconds attribution.
    """
    result = None
    tokens_in = tokens_out = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue  # garbage / partial line — tolerate
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "result":
            result = obj  # keep scanning; we want the last one
        elif obj.get("type") == "assistant":
            u = (obj.get("message") or {}).get("usage") or {}
            tokens_in += ((u.get("input_tokens") or 0)
                          + (u.get("cache_read_input_tokens") or 0)
                          + (u.get("cache_creation_input_tokens") or 0))
            tokens_out += (u.get("output_tokens") or 0)

    if result is None:
        return {
            "tokens_in": int(tokens_in), "tokens_out": int(tokens_out),
            "num_turns": 0, "is_error": False, "subtype": "", "api_usd": 0.0,
            "found_result": False,
        }
    return {
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "num_turns": int(result.get("num_turns") or 0),
        "is_error": bool(result.get("is_error", False)),
        "subtype": str(result.get("subtype") or ""),
        # total_cost_usd is real for hosted Claude models, 0 for the custom
        # ornith endpoint (no pricing) — which is correct: ornith has no API $.
        "api_usd": float(result.get("total_cost_usd") or 0.0),
        "found_result": True,
    }


def patch_target_files(patch_text: str) -> list:
    """Repo-relative paths a unified diff modifies (the `+++ b/<path>` side).
    Used to reset the hidden-test files to base before applying them, so the
    worker's edits to any test file are discarded (tests are authoritative)."""
    files = []
    for line in (patch_text or "").splitlines():
        if line.startswith("+++ "):
            p = line[4:].strip()
            if p == "/dev/null":
                continue
            if p.startswith("b/"):
                p = p[2:]
            files.append(p.split("\t")[0])
    return files


# --- stage → verdict table ------------------------------------------------
#
# A "stage" names *where* the trial resolved; classify() maps it to one of the
# four outcomes the harness understands. The table owns stage -> final-verdict
# ONLY; the retry/loop SEQUENCING (how many worker attempts, when to re-create a
# sandbox) lives in the caller (run_trial in modal_app.py), not here. The
# contract (from the runner spec):
#
#   worker_sandbox_create_failed  first attempt  -> retry
#                                 after retries  -> invalid   (detail="exhausted")
#   worker_no_result_line         first attempt  -> retry
#                                 after 1 retry  -> invalid   (detail="exhausted")
#   worker_reported_error         -> invalid   (CLI self-reported is_error)
#   base_sha_unresolved           -> invalid   (git rev-parse failed/empty)
#   scaffold_materialize_failed   -> invalid   (mkdir/tar/cp of .claude failed)
#   worker_diff_stage_failed      -> invalid   (git add of the worker change failed)
#   empty_diff                    -> fail      (worker produced no code change)
#   worker_timeout                -> PROCEED   (not terminal: caller still diffs
#                                               whatever the worker wrote so far)
#   worker_diff_apply_failed      -> invalid   (execution error, never "fail")
#   hidden_tests_apply_failed     -> invalid   (our patch, not the worker's fault)
#   verdict_sandbox_create_failed -> invalid   (Phase B sandbox create failed)
#   verify_exit_zero              -> pass
#   verify_exit_nonzero           -> fail
#   verify_timeout                -> fail       (a hung verify is a failed fix)
#
# "invalid" ≠ "fail": invalid is excluded from pass-rate denominators (PLAN
# "Error ≠ fail"); fail counts against the variant.

PROCEED = "proceed"

# Stages that are retryable on first sight and only become terminal (invalid)
# once the caller has exhausted its retry budget, signalled via detail.
_RETRY_STAGES = {"worker_sandbox_create_failed", "worker_no_result_line"}

_STAGE_VERDICT = {
    "worker_reported_error": "invalid",
    "base_sha_unresolved": "invalid",
    "scaffold_materialize_failed": "invalid",
    "worker_diff_stage_failed": "invalid",
    "empty_diff": "fail",
    "worker_timeout": PROCEED,
    "worker_diff_apply_failed": "invalid",
    "hidden_tests_apply_failed": "invalid",
    "verdict_sandbox_create_failed": "invalid",
    "verify_exit_zero": "pass",
    "verify_exit_nonzero": "fail",
    "verify_timeout": "fail",
}


def classify(stage: str, detail: str = "") -> str:
    """Map a stage name to 'pass'/'fail'/'invalid'/'retry' (or PROCEED).

    This is the stage -> final-verdict table ONLY. Retry/loop SEQUENCING (how
    many worker attempts, when the caller re-creates a sandbox vs. gives up)
    lives in the caller (run_trial), not here: the two retryable stages return
    'retry' until the caller passes detail='exhausted', at which point they
    resolve to 'invalid'. Unknown stages are treated as 'invalid' — an
    unclassified failure is an execution error, never a pass.
    """
    if stage in _RETRY_STAGES:
        return "invalid" if detail == "exhausted" else "retry"
    return _STAGE_VERDICT.get(stage, "invalid")


def worker_env(worker: dict, extra: dict | None = None) -> dict:
    """Environment for the `claude` exec inside the task sandbox.

    worker = {"model": str, "small_model": str, "base_url": str|None,
              "api_key": str (optional)}

    ANTHROPIC_BASE_URL is set only when base_url is truthy — the ornith path
    points Claude Code at the in-app LiteLLM proxy; the claude-* path omits it so
    the CLI talks to api.anthropic.com. The DISABLE_* / nonessential-traffic vars
    keep the worker hermetic (no autoupdate, telemetry, or error reporting that
    would add nondeterminism or egress). `extra` overrides base keys last.
    """
    env = {
        "ANTHROPIC_MODEL": worker["model"],
        "ANTHROPIC_SMALL_FAST_MODEL": worker.get("small_model") or worker["model"],
        "ANTHROPIC_API_KEY": worker.get("api_key") or "missing-key",
        # Claude Code refuses --dangerously-skip-permissions as root (the task
        # sandbox runs as root) unless IS_SANDBOX=1 — without it the CLI hangs
        # before emitting any output. The worker exec also feeds stdin from
        # /dev/null; an open non-TTY stdin is the other half of that hang.
        "IS_SANDBOX": "1",
        "DISABLE_AUTOUPDATER": "1",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
    if worker.get("base_url"):
        env["ANTHROPIC_BASE_URL"] = worker["base_url"]
    if extra:
        env.update(extra)
    return env


def node_claude_install_cmds() -> list[str]:
    """`run_commands` to layer Node 22 + the pinned Claude Code CLI onto a task
    image. Task images are Ubuntu 22.04 running as root with no Node; curl/ca
    certs may be absent (SWE-bench base images are minimal). Each command is one
    cached image layer.
    """
    return [
        "apt-get update && apt-get install -y --no-install-recommends "
        "curl ca-certificates gnupg",
        # NodeSource setup script adds the apt repo and refreshes the index.
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y --no-install-recommends nodejs",
        f"npm install -g @anthropic-ai/claude-code@{CLAUDE_CODE_VERSION}",
    ]
