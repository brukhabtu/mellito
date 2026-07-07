"""Unit tests for the pure trial-orchestration logic (no Modal, no GPU)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trial_logic as tl  # noqa: E402


def _result_line(**kw):
    base = {
        "type": "result", "subtype": "success", "is_error": False,
        "num_turns": 7, "total_cost_usd": 0.1234,
        "usage": {"input_tokens": 100, "output_tokens": 50,
                  "cache_read_input_tokens": 20, "cache_creation_input_tokens": 5},
    }
    base.update(kw)
    return json.dumps(base)


# --- parse_stream_json ----------------------------------------------------

def test_parse_normal_stream_sums_tokens_and_cost():
    lines = [
        json.dumps({"type": "system", "subtype": "init"}),
        json.dumps({"type": "assistant", "message": {"content": "hi"}}),
        _result_line(),
    ]
    u = tl.parse_stream_json(lines)
    assert u["found_result"] is True
    assert u["tokens_in"] == 100 + 20 + 5   # input + cache_read + cache_creation
    assert u["tokens_out"] == 50
    assert u["num_turns"] == 7
    assert u["is_error"] is False
    assert u["subtype"] == "success"
    assert abs(u["api_usd"] - 0.1234) < 1e-9


def test_parse_picks_last_result_line():
    lines = [_result_line(num_turns=1), _result_line(num_turns=9,
             usage={"input_tokens": 1, "output_tokens": 2})]
    u = tl.parse_stream_json(lines)
    assert u["num_turns"] == 9
    assert u["tokens_out"] == 2


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
        json.dumps({"type": "assistant"}),
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
              "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"):
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
