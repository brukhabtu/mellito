"""Unit tests for the P8 native-loop wrapper's pure decision logic.

Covers the per-session helpers factored into trial_logic (build_attempt_prompt,
should_stop_attempts, accumulate_usage) and the transcript-detection GLUE the
attempts loop relies on — that selection_analysis' worker-VERIFY detector, fed a
just-finished session's stream-json lines, classifies a self-verified session as
RAN_PASS (so should_stop_attempts stops the loop). No Modal, GPU, or network:
pure functions over inline synthetic stream-json, mirroring
test_trial_logic.py / test_selection_analysis.py style.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trial_logic as tl  # noqa: E402
from selection_analysis import detect_worker_verify  # noqa: E402


# --- build_attempt_prompt -------------------------------------------------

DESC = "Fix the off-by-one in foo().\n\nDetails: the loop overshoots by 1."


def test_attempt1_is_byte_identical_to_description():
    # The default single-attempt path MUST be a true no-op: attempt 1 returns
    # the description verbatim (byte-identical), for any n.
    assert tl.build_attempt_prompt(DESC, 1, 1) == DESC
    assert tl.build_attempt_prompt(DESC, 1, 3) == DESC
    # k<=0 is treated as attempt 1 too (defensive; never happens in run_trial).
    assert tl.build_attempt_prompt(DESC, 0, 3) == DESC


def test_attempt_k_gt_1_prepends_preamble_with_numbers_and_keeps_description():
    p2 = tl.build_attempt_prompt(DESC, 2, 3)
    assert p2 != DESC
    assert p2.startswith("[Attempt 2 of 3]")
    assert p2.endswith(DESC)                       # original description preserved
    assert "same workspace" in p2                  # persistence-channel language
    assert "git diff" in p2                        # review-your-own-work hint
    assert "VERIFY.txt" in p2                      # self-verify hint
    p3 = tl.build_attempt_prompt(DESC, 3, 3)
    assert p3.startswith("[Attempt 3 of 3]")


def test_preamble_absent_only_for_attempt1():
    # The "[Attempt" marker appears for every k>1 and NEVER for k==1.
    assert "[Attempt" not in tl.build_attempt_prompt(DESC, 1, 3)
    for k in (2, 3, 4, 5):
        assert f"[Attempt {k} of 5]" in tl.build_attempt_prompt(DESC, k, 5)


# --- should_stop_attempts --------------------------------------------------

def test_stop_only_on_ran_pass():
    assert tl.should_stop_attempts("RAN_PASS") is True
    assert tl.should_stop_attempts("RAN_FAIL") is False
    assert tl.should_stop_attempts("NEVER_RAN") is False
    assert tl.should_stop_attempts("") is False


# --- accumulate_usage ------------------------------------------------------

def test_accumulate_sums_counters_and_cost_across_sessions():
    acc = {"tokens_in": 0, "tokens_out": 0, "num_turns": 0, "api_usd": 0.0}
    tl.accumulate_usage(acc, {"tokens_in": 100, "tokens_out": 40,
                              "num_turns": 3, "api_usd": 0.0})
    tl.accumulate_usage(acc, {"tokens_in": 250, "tokens_out": 60,
                              "num_turns": 5, "api_usd": 0.0})
    assert acc["tokens_in"] == 350
    assert acc["tokens_out"] == 100
    assert acc["num_turns"] == 8
    assert acc["api_usd"] == 0.0


def test_accumulate_returns_same_dict_and_tolerates_missing_keys():
    acc = {"tokens_in": 0, "tokens_out": 0, "num_turns": 0, "api_usd": 0.0}
    ret = tl.accumulate_usage(acc, {"tokens_out": 7})  # missing keys default 0
    assert ret is acc
    assert acc["tokens_out"] == 7
    assert acc["tokens_in"] == 0 and acc["num_turns"] == 0
    # None values (parse_stream_json never emits them, but be robust) count 0.
    tl.accumulate_usage(acc, {"tokens_in": None, "tokens_out": None})
    assert acc["tokens_out"] == 7 and acc["tokens_in"] == 0


def test_accumulate_single_session_equals_that_session():
    # attempts=1 accounting: the folded acc equals the one session's usage for
    # exactly the fields _result reads.
    acc = {"tokens_in": 0, "tokens_out": 0, "num_turns": 0, "api_usd": 0.0}
    u = {"tokens_in": 11, "tokens_out": 22, "num_turns": 4, "api_usd": 0.0,
         "found_result": True, "is_error": False, "subtype": "success"}
    tl.accumulate_usage(acc, u)
    assert (acc["tokens_in"], acc["tokens_out"], acc["num_turns"]) == (11, 22, 4)


# --- transcript-detection glue (selection_analysis on a synthetic stream) ---
#
# The attempts loop calls detect_worker_verify on the JUST-FINISHED session's
# stream-json lines and stops iff state == RAN_PASS. Build a minimal stream the
# detector must classify RAN_PASS, and confirm should_stop_attempts agrees —
# the exact seam run_trial._stop_after exercises.

VERIFY_CMD = ("source /opt/miniconda3/bin/activate testbed && cd /testbed && "
              "pytest -rA astropy/modeling/tests/test_separable.py")


def _asst(mid, block):
    return json.dumps({"type": "assistant",
                       "message": {"id": mid, "role": "assistant",
                                   "content": [block],
                                   "usage": {"input_tokens": 1,
                                             "output_tokens": 1}}})


def _bash(mid, tid, command):
    return _asst(mid, {"type": "tool_use", "id": tid, "name": "Bash",
                       "input": {"command": command}})


def _tool_result(tid, content, is_error=None):
    return json.dumps({"type": "user",
                       "message": {"role": "user",
                                   "content": [{"type": "tool_result",
                                                "tool_use_id": tid,
                                                "content": content,
                                                "is_error": is_error}]}})


def _result_line():
    return json.dumps({"type": "result", "subtype": "success", "num_turns": 1,
                       "is_error": False})


def test_glue_ran_pass_stream_triggers_stop():
    # A session that ran VERIFY and whose last such run returned non-error ->
    # RAN_PASS -> the loop stops.
    lines = [
        json.dumps({"type": "system", "subtype": "init"}),
        _bash("A", "t1", "pytest -rA astropy/modeling/tests/test_separable.py"),
        _tool_result("t1", "5 passed", is_error=False),
        _result_line(),
    ]
    state = detect_worker_verify(lines, VERIFY_CMD)["state"]
    assert state == "RAN_PASS"
    assert tl.should_stop_attempts(state) is True


def test_glue_ran_fail_stream_continues():
    lines = [
        json.dumps({"type": "system", "subtype": "init"}),
        _bash("A", "t1", "pytest -rA astropy/modeling/tests/test_separable.py"),
        _tool_result("t1", "1 failed", is_error=True),
        _result_line(),
    ]
    state = detect_worker_verify(lines, VERIFY_CMD)["state"]
    assert state == "RAN_FAIL"
    assert tl.should_stop_attempts(state) is False


def test_glue_never_ran_stream_continues():
    lines = [
        json.dumps({"type": "system", "subtype": "init"}),
        _bash("A", "t1", "ls /testbed"),
        _tool_result("t1", "setup.py", is_error=False),
        _result_line(),
    ]
    state = detect_worker_verify(lines, VERIFY_CMD)["state"]
    assert state == "NEVER_RAN"
    assert tl.should_stop_attempts(state) is False
