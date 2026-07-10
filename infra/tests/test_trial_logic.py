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
    w = {"model": "other-model", "small_model": "other-small-model",
         "base_url": None}
    env = tl.worker_env(w)
    assert "ANTHROPIC_BASE_URL" not in env
    assert env["ANTHROPIC_API_KEY"] == "missing-key"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "other-small-model"
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


# --- detect_native_verify (P9-F native + attempts self-VERIFY detection) ---
#
# Synthetic native_driver transcripts: assistant rows carry OpenAI `tool_calls`
# (function.name=="bash", arguments a JSON string with "command"); role:"tool"
# reply rows carry the command output ALWAYS ending in "[exit code: N]" (or a
# timeout marker); a terminal native_result row closes the run. The detector is
# the native analogue of selection_analysis.detect_worker_verify and feeds the
# SAME should_stop_attempts.

NATIVE_VERIFY = ("source /opt/miniconda3/bin/activate testbed && cd /testbed && "
                 "pytest -rA astropy/modeling/tests/test_separable.py")


def _n_tc(command, call_id="c1", name="bash", raw_args=None):
    """One OpenAI-shape bash tool_call (arguments as the JSON-string wire form)."""
    args = raw_args if raw_args is not None else json.dumps({"command": command})
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": args}}


def _n_asst(*tool_calls, content=""):
    row = {"role": "assistant", "content": content}
    if tool_calls:
        row["tool_calls"] = list(tool_calls)
    return json.dumps(row)


def _n_tool(content, call_id="c1"):
    return json.dumps({"role": "tool", "tool_call_id": call_id, "content": content})


def _n_result(ended="done", turns=1):
    return json.dumps({"type": "native_result", "turns": turns, "ended": ended,
                       "usage_total": {"tokens_in": 1, "tokens_out": 1}})


def _n_exit(code):
    """A tool reply body ending in native_driver's trailing exit-code marker."""
    return f"...ran the tests...\n[exit code: {code}]"


def _state(rows, verify=NATIVE_VERIFY):
    return tl.detect_native_verify("\n".join(rows), verify)["state"]


def test_native_verify_pass_substring_exit0():
    # (a) the whole verify command appears in the run command; reply exits 0.
    rows = [
        json.dumps({"role": "system", "content": "s"}),
        json.dumps({"role": "user", "content": "Begin."}),
        _n_asst(_n_tc(NATIVE_VERIFY, "t1")),
        _n_tool(_n_exit(0), "t1"),
        _n_result("done"),
    ]
    assert _state(rows) == "RAN_PASS"
    assert tl.should_stop_attempts(_state(rows)) is True   # loop would stop


def test_native_verify_fail_exit_nonzero():
    rows = [
        _n_asst(_n_tc(NATIVE_VERIFY, "t1")),
        _n_tool(_n_exit(1), "t1"),
        _n_result("done"),
    ]
    assert _state(rows) == "RAN_FAIL"
    assert tl.should_stop_attempts(_state(rows)) is False   # loop would continue


def test_native_verify_never_ran():
    rows = [
        _n_asst(_n_tc("ls /testbed", "t1")),
        _n_tool(_n_exit(0), "t1"),
        _n_asst(_n_tc("git status", "t2")),
        _n_tool(_n_exit(0), "t2"),
        _n_result("done"),
    ]
    assert _state(rows) == "NEVER_RAN"


def test_native_verify_cat_only_read_does_not_qualify():
    # A command that merely READS VERIFY.txt (exit 0) must NOT read as RAN_PASS —
    # the deliberately-stricter-than-CC edge from the P9-F pre-registration.
    for read_cmd in ("cat VERIFY.txt", "cat /testbed/VERIFY.txt",
                     "head -50 /testbed/VERIFY.txt", "less VERIFY.txt",
                     "grep pytest /testbed/VERIFY.txt"):
        rows = [
            _n_asst(_n_tc(read_cmd, "t1")),
            _n_tool(_n_exit(0), "t1"),   # cat exits 0 — would be RAN_PASS if it qualified
            _n_result("done"),
        ]
        assert _state(rows) == "NEVER_RAN", read_cmd


def test_native_verify_txt_execution_patterns_qualify():
    # (b) running VERIFY.txt (not reading it) qualifies. verify_cmd="" isolates
    # this path from the substring path (a).
    for exec_cmd in ("bash /testbed/VERIFY.txt", "sh VERIFY.txt",
                     "source /testbed/VERIFY.txt", ". /testbed/VERIFY.txt",
                     'eval "$(cat /testbed/VERIFY.txt)"',
                     'bash -c "$(cat /testbed/VERIFY.txt)"'):
        rows = [
            _n_asst(_n_tc(exec_cmd, "t1")),
            _n_tool(_n_exit(0), "t1"),
            _n_result("done"),
        ]
        assert tl.detect_native_verify("\n".join(rows), "")["state"] == "RAN_PASS", exec_cmd
    # Empty verify_cmd must NOT blanket-match a plain command.
    rows = [_n_asst(_n_tc("ls", "t1")), _n_tool(_n_exit(0), "t1"), _n_result("done")]
    assert tl.detect_native_verify("\n".join(rows), "")["state"] == "NEVER_RAN"


def test_native_verify_last_invocation_wins():
    fail_then_pass = [
        _n_asst(_n_tc(NATIVE_VERIFY, "t1")),
        _n_tool(_n_exit(1), "t1"),
        _n_asst(_n_tc(NATIVE_VERIFY, "t2")),
        _n_tool(_n_exit(0), "t2"),
        _n_result("done"),
    ]
    assert _state(fail_then_pass) == "RAN_PASS"
    pass_then_fail = [
        _n_asst(_n_tc(NATIVE_VERIFY, "t1")),
        _n_tool(_n_exit(0), "t1"),
        _n_asst(_n_tc(NATIVE_VERIFY, "t2")),
        _n_tool(_n_exit(1), "t2"),
        _n_result("done"),
    ]
    assert _state(pass_then_fail) == "RAN_FAIL"


def test_native_verify_timeout_marker_is_fail():
    rows = [
        _n_asst(_n_tc(NATIVE_VERIFY, "t1")),
        _n_tool("partial output before it hung\n[command timed out after 120s]", "t1"),
        _n_result("done"),
    ]
    assert _state(rows) == "RAN_FAIL"


def test_native_verify_missing_tool_row_is_fail():
    # Ran VERIFY but the driver died before writing the tool reply (next row is
    # the native_result) -> no exit-0 signal -> fail.
    rows = [_n_asst(_n_tc(NATIVE_VERIFY, "t1")), _n_result("error")]
    assert _state(rows) == "RAN_FAIL"
    # Also: the next row is another assistant, so the VERIFY call has no reply.
    rows2 = [
        _n_asst(_n_tc(NATIVE_VERIFY, "t1")),
        _n_asst(_n_tc("echo done", "t2")),
        _n_tool(_n_exit(0), "t2"),
        _n_result("done"),
    ]
    assert _state(rows2) == "RAN_FAIL"


def test_native_verify_only_first_tool_call_considered():
    # native_driver executes ONLY tool_calls[0]; the detector must match that.
    first_is_verify = [
        _n_asst(_n_tc(NATIVE_VERIFY, "t1"), _n_tc("echo hi", "t2")),
        _n_tool(_n_exit(0), "t1"),   # reply is the first call's output
        _n_result("done"),
    ]
    assert _state(first_is_verify) == "RAN_PASS"
    # A verify command that is only the SECOND call is never executed -> ignored.
    second_is_verify = [
        _n_asst(_n_tc("echo hi", "t1"), _n_tc(NATIVE_VERIFY, "t2")),
        _n_tool(_n_exit(0), "t1"),
        _n_result("done"),
    ]
    assert _state(second_is_verify) == "NEVER_RAN"


def test_native_verify_malformed_arguments_tolerated():
    # A tool_call with non-JSON arguments is skipped, never raised; a later
    # well-formed VERIFY run still classifies.
    rows = [
        _n_asst(_n_tc("", "t1", raw_args="{not valid json")),
        _n_tool(_n_exit(0), "t1"),
        _n_asst(_n_tc(NATIVE_VERIFY, "t2")),
        _n_tool(_n_exit(0), "t2"),
        _n_result("done"),
    ]
    assert _state(rows) == "RAN_PASS"
    # If the only verify-looking call has malformed args, it can't qualify.
    rows2 = [
        _n_asst(_n_tc("", "t1", raw_args="{broken")),
        _n_tool(_n_exit(0), "t1"),
        _n_result("done"),
    ]
    assert _state(rows2) == "NEVER_RAN"


def test_native_verify_non_bash_first_call_skipped():
    # A non-bash first tool_call would get an error reply from the driver, never
    # running VERIFY -> not a qualifying invocation.
    rows = [
        _n_asst(_n_tc("whatever", "t1", name="python")),
        _n_tool(_n_exit(0), "t1"),
        _n_result("done"),
    ]
    assert _state(rows) == "NEVER_RAN"


def test_native_verify_tolerates_garbage_and_accepts_list_input():
    rows = [
        "not json at all",
        "",
        "{partial json",
        _n_asst(_n_tc(NATIVE_VERIFY, "t1")),
        _n_tool(_n_exit(0), "t1"),
        _n_result("done"),
    ]
    # list-of-lines input and text input both work; garbage lines are skipped.
    assert tl.detect_native_verify(rows, NATIVE_VERIFY)["state"] == "RAN_PASS"
    assert _state(rows) == "RAN_PASS"


def test_native_verify_multiline_command_normalized():
    # A heredoc / multiline run command still matches the (collapsed) verify cmd.
    multiline = ("source /opt/miniconda3/bin/activate testbed &&\n"
                 "  cd /testbed &&\n"
                 "  pytest -rA astropy/modeling/tests/test_separable.py")
    rows = [
        _n_asst(_n_tc(multiline, "t1")),
        _n_tool(_n_exit(0), "t1"),
        _n_result("done"),
    ]
    assert _state(rows) == "RAN_PASS"


def test_native_verify_exit_code_10_is_fail_not_prefix_match():
    # A trailing "[exit code: 10]" must NOT be read as a 0-exit (no false pass).
    rows = [
        _n_asst(_n_tc(NATIVE_VERIFY, "t1")),
        _n_tool("boom\n[exit code: 10]", "t1"),
        _n_result("done"),
    ]
    assert _state(rows) == "RAN_FAIL"
