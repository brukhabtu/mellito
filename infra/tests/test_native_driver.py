"""Unit tests for the P9-E native minimal-loop driver (no network, no sandbox).

Covers the pure pieces: shell-block extraction / turn classification, output
truncation, the malformed-turn state machine, transcript row shapes, one full
loop driven by a fake chat+exec client, and the run_trial-side transcript
readback (trial_logic.parse_native_transcript).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import native_driver as nd  # noqa: E402
import trial_logic as tl  # noqa: E402


# --- fakes ----------------------------------------------------------------

class FakeChat:
    """Scripted chat_fn: returns the queued responses in order; records the
    messages it was called with. Raises IndexError if over-drawn (surfaces a
    loop that ran longer than the test scripted)."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append([dict(m) for m in messages])
        return self.responses.pop(0)


def _resp(content, reasoning="", usage=None):
    return {"content": content, "reasoning": reasoning, "usage": usage or {}}


def _collect_emit():
    rows = []
    return rows, rows.append


def _fake_now(seq):
    """A now() callable yielding `seq` then repeating its last value forever."""
    box = {"i": 0}

    def now():
        i = box["i"]
        if i < len(seq):
            box["i"] += 1
            return seq[i]
        return seq[-1]

    return now


def _bash(cmd):
    return f"Let me run this.\n```bash\n{cmd}\n```"


# --- extraction / classification ------------------------------------------

def test_extract_single_block():
    assert nd.extract_bash_blocks(_bash("ls -la")) == ["ls -la"]


def test_classify_single_command():
    info = nd.classify_turn(_bash("pytest -q"))
    assert info["kind"] == "command"
    assert info["command"] == "pytest -q"
    assert info["multiple_blocks"] is False


def test_classify_multiple_blocks_takes_first_and_flags():
    text = _bash("first cmd") + "\nthen\n" + _bash("second cmd")
    info = nd.classify_turn(text)
    assert info["kind"] == "command"
    assert info["command"] == "first cmd"          # first only
    assert info["multiple_blocks"] is True         # protocol-violation flag


def test_classify_done_line():
    info = nd.classify_turn("The fix verifies.\nDONE")
    assert info["kind"] == "done"
    assert info["command"] is None


def test_classify_command_wins_over_done():
    # A command present means the model is not finished — it beats a stray DONE.
    info = nd.classify_turn("DONE\n" + _bash("echo still going"))
    assert info["kind"] == "command"
    assert info["command"] == "echo still going"


def test_classify_malformed_prose_only():
    info = nd.classify_turn("I think we should edit the parser somehow.")
    assert info["kind"] == "malformed"


def test_classify_bare_fence_is_not_a_command():
    # A bare ``` fence (no shell language) is NOT executed — off protocol.
    info = nd.classify_turn("```\nls\n```")
    assert info["kind"] == "malformed"


def test_classify_accepts_sh_and_shell_fences():
    assert nd.classify_turn("```sh\nls\n```")["kind"] == "command"
    assert nd.classify_turn("```shell\nls\n```")["kind"] == "command"


def test_done_requires_line_by_itself():
    # "DONE" embedded in prose is not the terminator.
    info = nd.classify_turn("I am not DONE yet, more to do.")
    assert info["kind"] == "malformed"


# --- truncation -----------------------------------------------------------

def test_truncate_passthrough_when_short():
    assert nd.truncate_output("hello", limit=100) == "hello"


def test_truncate_keeps_head_and_tail():
    text = "A" * 5000 + "B" * 5000
    out = nd.truncate_output(text, limit=1000)
    assert len(out) < len(text)
    assert out.startswith("A")
    assert out.endswith("B")           # tail preserved (errors surface at end)
    assert "truncated" in out


# --- system prompt is frozen / minimal ------------------------------------

def test_system_prompt_interpolates_task_and_names_verify():
    sp = nd.build_system_prompt("Fix the flaky test in foo.py")
    assert "Fix the flaky test in foo.py" in sp
    assert "VERIFY.txt" in sp
    assert "/testbed" in sp
    assert "```bash" in sp
    assert "DONE" in sp
    # Control arm: no completion-contract / self-direction coaching.
    assert sp.count("\n") < 20


# --- full loop: command -> output -> DONE ---------------------------------

def test_loop_command_then_done():
    chat = FakeChat([
        _resp(_bash("cat VERIFY.txt"), reasoning="<think>plan</think>",
              usage={"prompt_tokens": 100, "completion_tokens": 20}),
        _resp("The fix passes.\nDONE",
              usage={"prompt_tokens": 150, "completion_tokens": 8}),
    ])
    executed = []

    def fake_exec(cmd):
        executed.append(cmd)
        return "pytest ... 1 passed\n[exit code: 0]"

    rows, emit = _collect_emit()
    out = nd.run_loop(chat, fake_exec, "Do the task", emit=emit, timeout_s=10_000)

    assert out["ended"] == "done"
    assert out["turns"] == 1
    assert executed == ["cat VERIFY.txt"]
    assert out["totals"] == {"tokens_in": 250, "tokens_out": 28}

    # Row order: system, kickoff-user, assistant(cmd), user(output),
    # assistant(DONE), native_result.
    assert rows[0]["role"] == "system"
    assert rows[1] == {"role": "user", "content": nd.KICKOFF}
    assert rows[2]["role"] == "assistant"
    assert rows[2]["reasoning"] == "<think>plan</think>"   # reasoning recorded
    assert rows[2]["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}
    assert rows[3] == {"role": "user",
                       "content": "pytest ... 1 passed\n[exit code: 0]"}
    assert rows[4]["role"] == "assistant"
    assert rows[-1] == {"type": "native_result", "turns": 1, "ended": "done",
                        "usage_total": {"tokens_in": 250, "tokens_out": 28}}

    # Reasoning is NEVER fed back: the 2nd chat call's message list carries the
    # assistant turn WITHOUT a reasoning field.
    second_call_msgs = chat.calls[1]
    asst = [m for m in second_call_msgs if m["role"] == "assistant"]
    assert asst and "reasoning" not in asst[0]
    assert asst[0]["content"] == _bash("cat VERIFY.txt")


# --- malformed-turn policy state machine ----------------------------------

def test_two_consecutive_malformed_ends_protocol():
    chat = FakeChat([_resp("no command here"), _resp("still no command")])
    rows, emit = _collect_emit()
    out = nd.run_loop(chat, lambda c: "x", "task", emit=emit, timeout_s=10_000)
    assert out["ended"] == "protocol"
    assert out["turns"] == 0
    # Exactly one reminder was emitted (after the first malformed turn).
    reminders = [r for r in rows
                 if r.get("role") == "user" and r.get("content") == nd.PROTOCOL_REMINDER]
    assert len(reminders) == 1


def test_malformed_then_command_resets_counter():
    chat = FakeChat([
        _resp("no command"),                 # malformed #1 -> reminder
        _resp(_bash("echo hi")),             # command -> resets
        _resp("prose again"),                # malformed #1 again (reset worked)
        _resp("still prose"),                # malformed #2 -> protocol end
    ])
    rows, emit = _collect_emit()
    out = nd.run_loop(chat, lambda c: "out", "task", emit=emit, timeout_s=10_000)
    assert out["ended"] == "protocol"
    assert out["turns"] == 1                  # the one command ran
    reminders = [r for r in rows
                 if r.get("role") == "user" and r.get("content") == nd.PROTOCOL_REMINDER]
    assert len(reminders) == 2                # one before reset, one after


# --- limits: max_turns and wall-clock timeout -----------------------------

def test_max_turns_ends_loop():
    chat = FakeChat([_resp(_bash("echo x")) for _ in range(5)])
    rows, emit = _collect_emit()
    out = nd.run_loop(chat, lambda c: "out", "task", emit=emit,
                      timeout_s=10_000, max_turns=3)
    assert out["ended"] == "max_turns"
    assert out["turns"] == 3
    assert len(chat.calls) == 3               # stopped exactly at the cap


def test_wall_clock_timeout_ends_loop():
    chat = FakeChat([_resp(_bash("echo x")), _resp(_bash("echo y"))])
    # now(): start=0, iter1 check=1 (<100 ok), iter2 check=9999 (>=100 -> timeout).
    now = _fake_now([0, 1, 9999])
    rows, emit = _collect_emit()
    out = nd.run_loop(chat, lambda c: "out", "task", emit=emit,
                      timeout_s=100, now=now)
    assert out["ended"] == "timeout"
    assert out["turns"] == 1
    assert rows[-1]["ended"] == "timeout"


def test_chat_exception_ends_error_and_still_writes_result():
    def boom(messages):
        raise RuntimeError("connection reset")

    rows, emit = _collect_emit()
    out = nd.run_loop(boom, lambda c: "out", "task", emit=emit, timeout_s=10_000)
    assert out["ended"] == "error"
    assert rows[-1]["type"] == "native_result"
    assert rows[-1]["ended"] == "error"


def test_multiple_blocks_records_violation_row():
    text = _bash("first") + "\n" + _bash("second")
    chat = FakeChat([_resp(text), _resp("DONE")])
    rows, emit = _collect_emit()
    nd.run_loop(chat, lambda c: "out", "task", emit=emit, timeout_s=10_000)
    asst = [r for r in rows if r.get("role") == "assistant"]
    assert asst[0].get("protocol_violation") == "multiple_bash_blocks"


# --- run_trial-side readback (trial_logic.parse_native_transcript) --------

def test_parse_native_transcript_uses_result_total():
    rows, emit = _collect_emit()
    chat = FakeChat([
        _resp(_bash("ls"), usage={"prompt_tokens": 100, "completion_tokens": 20}),
        _resp("DONE", usage={"prompt_tokens": 130, "completion_tokens": 5}),
    ])
    nd.run_loop(chat, lambda c: "out", "task", emit=emit, timeout_s=10_000)
    text = "\n".join(json.dumps(r) for r in rows)
    u = tl.parse_native_transcript(text)
    assert u["found_result"] is True
    assert u["ended"] == "done"
    assert u["num_turns"] == 1
    assert u["tokens_in"] == 230
    assert u["tokens_out"] == 25


def test_parse_native_transcript_no_result_line_falls_back():
    # Assistant rows but no native_result (driver died before finishing).
    lines = [
        json.dumps({"role": "system", "content": "s"}),
        json.dumps({"role": "assistant", "content": "x",
                    "usage": {"prompt_tokens": 40, "completion_tokens": 7}}),
    ]
    u = tl.parse_native_transcript(lines)
    assert u["found_result"] is False
    assert u["num_turns"] == 0
    assert u["tokens_in"] == 40
    assert u["tokens_out"] == 7


def test_parse_native_transcript_tolerates_garbage():
    text = "not json\n\n{partial\n" + json.dumps(
        {"type": "native_result", "turns": 3, "ended": "max_turns",
         "usage_total": {"tokens_in": 5, "tokens_out": 6}})
    u = tl.parse_native_transcript(text)
    assert u["found_result"] is True
    assert u["num_turns"] == 3
    assert u["ended"] == "max_turns"
    assert u["tokens_in"] == 5 and u["tokens_out"] == 6
