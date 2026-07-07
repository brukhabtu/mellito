"""Unit tests for the pure trial-orchestration logic (no Modal, no GPU)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trial_logic as tl  # noqa: E402


def _result_line(**kw):
    # NOTE: the result line's own usage is intentionally all-zero, mirroring
    # what Claude Code emits against the custom (unpriced) ornith endpoint —
    # tokens must come from the assistant messages, not here.
    base = {
        "type": "result", "subtype": "success", "is_error": False,
        "num_turns": 7, "total_cost_usd": 0.1234,
        "usage": {"input_tokens": 0, "output_tokens": 0,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    }
    base.update(kw)
    return json.dumps(base)


def _asst(inp=0, out=0, cr=0, cc=0):
    return json.dumps({"type": "assistant", "message": {"usage": {
        "input_tokens": inp, "output_tokens": out,
        "cache_read_input_tokens": cr, "cache_creation_input_tokens": cc}}})


# --- parse_stream_json ----------------------------------------------------

def test_parse_sums_tokens_from_assistant_messages():
    # Tokens come from per-assistant usage (summed over turns), NOT the
    # all-zero result line — the ornith-endpoint reality.
    lines = [
        json.dumps({"type": "system", "subtype": "init"}),
        _asst(inp=100, out=50, cr=20, cc=5),
        _asst(inp=200, out=30, cr=10, cc=0),
        _result_line(),
    ]
    u = tl.parse_stream_json(lines)
    assert u["found_result"] is True
    assert u["tokens_in"] == (100 + 20 + 5) + (200 + 10 + 0)
    assert u["tokens_out"] == 50 + 30
    assert u["num_turns"] == 7
    assert u["is_error"] is False
    assert u["subtype"] == "success"
    assert abs(u["api_usd"] - 0.1234) < 1e-9


def test_parse_tokens_summed_even_without_result_line():
    u = tl.parse_stream_json([_asst(inp=10, out=7)])
    assert u["found_result"] is False
    assert u["tokens_out"] == 7
    assert u["tokens_in"] == 10


def test_parse_picks_last_result_line():
    lines = [_result_line(num_turns=1), _result_line(num_turns=9)]
    u = tl.parse_stream_json(lines)
    assert u["num_turns"] == 9


def test_patch_target_files():
    patch = ("diff --git a/tests/x_test.py b/tests/x_test.py\n"
             "--- a/tests/x_test.py\n+++ b/tests/x_test.py\n@@ -1 +1 @@\n-a\n+b\n"
             "diff --git a/pkg/new.py b/pkg/new.py\n"
             "--- /dev/null\n+++ b/pkg/new.py\n@@ -0,0 +1 @@\n+x\n")
    assert tl.patch_target_files(patch) == ["tests/x_test.py", "pkg/new.py"]
    assert tl.patch_target_files("") == []


def test_parse_is_error_result():
    lines = [_result_line(is_error=True, subtype="error_max_turns",
                          total_cost_usd=0.9)]
    u = tl.parse_stream_json(lines)
    assert u["found_result"] is True
    assert u["is_error"] is True
    assert u["subtype"] == "error_max_turns"
    assert abs(u["api_usd"] - 0.9) < 1e-9


def test_parse_no_result_line():
    lines = [json.dumps({"type": "assistant", "message": {"content": "x"}}),
             json.dumps({"type": "system"})]
    u = tl.parse_stream_json(lines)
    assert u["found_result"] is False
    assert u["tokens_in"] == 0 and u["tokens_out"] == 0
    assert u["api_usd"] == 0.0
    assert u["num_turns"] == 0


def test_parse_tolerates_garbage_interleaved():
    lines = [
        "not json at all",
        "",
        "{partial json",
        _asst(out=50),
        _result_line(),
        "\x00\x01 trailing noise",
    ]
    u = tl.parse_stream_json(lines)
    assert u["found_result"] is True
    assert u["tokens_out"] == 50


def test_parse_missing_usage_defaults_zero():
    lines = [json.dumps({"type": "result", "subtype": "ok", "is_error": False})]
    u = tl.parse_stream_json(lines)
    assert u["found_result"] is True
    assert u["tokens_in"] == 0 and u["tokens_out"] == 0
    assert u["api_usd"] == 0.0


# --- classify -------------------------------------------------------------

def test_classify_terminal_stages():
    assert tl.classify("empty_diff") == "fail"
    assert tl.classify("worker_diff_apply_failed") == "invalid"
    assert tl.classify("hidden_tests_apply_failed") == "invalid"
    assert tl.classify("verify_exit_zero") == "pass"
    assert tl.classify("verify_exit_nonzero") == "fail"
    assert tl.classify("verify_timeout") == "fail"


def test_classify_new_invalid_stages():
    # Stages added in the review pass: each is an execution error, never a fail.
    for stage in ("worker_reported_error", "base_sha_unresolved",
                  "scaffold_materialize_failed", "worker_diff_stage_failed",
                  "verdict_sandbox_create_failed"):
        assert tl.classify(stage) == "invalid", stage


def test_classify_worker_timeout_proceeds():
    assert tl.classify("worker_timeout") == tl.PROCEED
    assert tl.PROCEED == "proceed"


def test_classify_retry_stages():
    assert tl.classify("worker_sandbox_create_failed") == "retry"
    assert tl.classify("worker_sandbox_create_failed", "exhausted") == "invalid"
    assert tl.classify("worker_no_result_line") == "retry"
    assert tl.classify("worker_no_result_line", "exhausted") == "invalid"


def test_classify_unknown_stage_is_invalid():
    assert tl.classify("something_unexpected") == "invalid"


# --- worker_env -----------------------------------------------------------

def test_worker_env_ornith_sets_base_url_and_key():
    w = {"model": "ornith-35b", "small_model": "ornith-35b",
         "base_url": "https://proxy.example/", "api_key": "sk-proxy-real"}
    env = tl.worker_env(w)
    assert env["ANTHROPIC_BASE_URL"] == "https://proxy.example/"
    assert env["ANTHROPIC_MODEL"] == "ornith-35b"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "ornith-35b"
    assert env["ANTHROPIC_API_KEY"] == "sk-proxy-real"
    for k in ("DISABLE_AUTOUPDATER", "DISABLE_TELEMETRY", "DISABLE_ERROR_REPORTING",
              "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "IS_SANDBOX"):
        assert env[k] == "1"


def test_worker_env_missing_key_falls_back_to_sentinel():
    # No api_key (ornith path before run_trial fills it) or api_key=None -> the
    # 'missing-key' sentinel, NOT any hardcoded proxy key.
    w = {"model": "claude-sonnet-4-5", "small_model": "claude-haiku-4-5-20251001",
         "base_url": None}
    env = tl.worker_env(w)
    assert "ANTHROPIC_BASE_URL" not in env
    assert env["ANTHROPIC_API_KEY"] == "missing-key"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "claude-haiku-4-5-20251001"
    env2 = tl.worker_env({"model": "ornith-35b", "base_url": "https://p/",
                          "api_key": None})
    assert env2["ANTHROPIC_API_KEY"] == "missing-key"


def test_worker_env_extra_overrides():
    w = {"model": "m", "small_model": "s", "base_url": None}
    env = tl.worker_env(w, extra={"ANTHROPIC_API_KEY": "sk-real", "X": "1"})
    assert env["ANTHROPIC_API_KEY"] == "sk-real"
    assert env["X"] == "1"


def test_worker_env_small_model_defaults_to_model():
    env = tl.worker_env({"model": "m", "base_url": None})
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "m"


# --- install cmds ---------------------------------------------------------

def test_node_claude_install_cmds_pins_version():
    cmds = tl.node_claude_install_cmds()
    assert any("setup_22.x" in c for c in cmds)
    assert any(f"@anthropic-ai/claude-code@{tl.CLAUDE_CODE_VERSION}" in c for c in cmds)
    assert tl.CLAUDE_CODE_VERSION == "2.0.14"
