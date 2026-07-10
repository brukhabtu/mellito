"""Unit tests for the P9-E native minimal-loop driver (no network, no sandbox).

Revision 2 protocol: OpenAI tools API transport (the model's native qwen3_xml
tool format, parsed server-side). Covers the pure pieces: tool-call command
extraction, the first-only rule for multiple tool_calls, no-tool-call = done,
output truncation, transcript row shapes, one full loop driven by a fake
chat+exec client, and the run_trial-side transcript readback
(trial_logic.parse_native_transcript).
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


def _tc(command, call_id="call_1", name="bash", raw_args=None):
    """One OpenAI-shape tool_call (arguments as a JSON string, the wire form)."""
    args = raw_args if raw_args is not None else json.dumps({"command": command})
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": args}}


def _resp(content="", tool_calls=None, reasoning="", usage=None):
    return {"content": content, "reasoning": reasoning,
            "usage": usage or {}, "tool_calls": tool_calls or []}


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


# --- extract_command --------------------------------------------------------

def test_extract_command_wire_shape():
    cmd, err = nd.extract_command(_tc("pytest -q"))
    assert cmd == "pytest -q" and err is None


def test_extract_command_already_parsed_args():
    tc = {"id": "c", "function": {"name": "bash",
                                  "arguments": {"command": "ls -la"}}}
    cmd, err = nd.extract_command(tc)
    assert cmd == "ls -la" and err is None


def test_extract_command_bad_json_args():
    cmd, err = nd.extract_command(_tc("", raw_args="{not json"))
    assert cmd is None and "JSON" in err


def test_extract_command_missing_command_key():
    cmd, err = nd.extract_command(_tc("", raw_args=json.dumps({"cmd": "ls"})))
    assert cmd is None and "command" in err


def test_extract_command_empty_command_rejected():
    cmd, err = nd.extract_command(_tc("   "))
    assert cmd is None and err


def test_extract_command_unknown_tool():
    cmd, err = nd.extract_command(_tc("ls", name="python"))
    assert cmd is None and "unknown tool" in err


# --- truncation -------------------------------------------------------------

def test_truncate_passthrough_when_short():
    assert nd.truncate_output("hello", limit=100) == "hello"


def test_truncate_keeps_head_and_tail():
    text = "A" * 5000 + "B" * 5000
    out = nd.truncate_output(text, limit=1000)
    assert len(out) < len(text)
    assert out.startswith("A")
    assert out.endswith("B")           # tail preserved (errors surface at end)
    assert "truncated" in out


# --- system prompt is frozen / minimal --------------------------------------

def test_system_prompt_interpolates_task_and_names_verify():
    sp = nd.build_system_prompt("Fix the flaky test in foo.py")
    assert "Fix the flaky test in foo.py" in sp
    assert "VERIFY.txt" in sp
    assert "/testbed" in sp
    assert "bash tool" in sp
    assert "reply without\ncalling any tool" in sp.replace("\r\n", "\n")
    # Control arm: no completion-contract / self-direction coaching, and none
    # of the retired fence-protocol text.
    assert "```" not in sp
    assert "DONE" not in sp
    assert sp.count("\n") < 20


def test_bash_tool_schema_frozen():
    fn = nd.BASH_TOOL["function"]
    assert nd.BASH_TOOL["type"] == "function"
    assert fn["name"] == "bash"
    assert fn["parameters"]["required"] == ["command"]
    assert list(fn["parameters"]["properties"]) == ["command"]


# --- full loop: tool call -> output -> no-tool-call done --------------------

def test_loop_command_then_done():
    chat = FakeChat([
        _resp(content="Checking the verify command.",
              tool_calls=[_tc("cat VERIFY.txt", call_id="call_A")],
              reasoning="<think>plan</think>",
              usage={"prompt_tokens": 100, "completion_tokens": 20}),
        _resp(content="The fix passes. All done.",
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

    # Row order: system, kickoff-user, assistant(tool_call), tool(output),
    # assistant(done), native_result.
    assert rows[0]["role"] == "system"
    assert rows[1] == {"role": "user", "content": nd.KICKOFF}
    assert rows[2]["role"] == "assistant"
    assert rows[2]["tool_calls"][0]["id"] == "call_A"
    assert rows[2]["reasoning"] == "<think>plan</think>"   # reasoning recorded
    assert rows[2]["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}
    assert "protocol_violation" not in rows[2]
    assert rows[3] == {"role": "tool", "tool_call_id": "call_A",
                       "content": "pytest ... 1 passed\n[exit code: 0]"}
    assert rows[4]["role"] == "assistant"
    assert "tool_calls" not in rows[4]
    assert rows[-1] == {"type": "native_result", "turns": 1, "ended": "done",
                        "usage_total": {"tokens_in": 250, "tokens_out": 28}}

    # Standard OpenAI multi-turn tool flow in the 2nd call's history:
    # assistant WITH tool_calls, then the matching role:"tool" reply.
    msgs = chat.calls[1]
    asst = [m for m in msgs if m["role"] == "assistant"]
    assert asst and asst[0]["tool_calls"][0]["id"] == "call_A"
    assert "reasoning" not in asst[0]          # reasoning NEVER fed back
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert tool_msgs == [{"role": "tool", "tool_call_id": "call_A",
                          "content": "pytest ... 1 passed\n[exit code: 0]"}]


def test_no_tool_call_first_reply_is_done_not_malformed():
    chat = FakeChat([_resp(content="This task needs no changes; summary here.")])
    rows, emit = _collect_emit()
    out = nd.run_loop(chat, lambda c: "x", "task", emit=emit, timeout_s=10_000)
    assert out["ended"] == "done"
    assert out["turns"] == 0
    assert rows[-1]["ended"] == "done"


# --- multiple tool_calls: first only + violation flag ------------------------

def test_multiple_tool_calls_first_only_and_flagged():
    chat = FakeChat([
        _resp(tool_calls=[_tc("first cmd", call_id="c1"),
                          _tc("second cmd", call_id="c2")]),
        _resp(content="done"),
    ])
    executed = []
    rows, emit = _collect_emit()
    out = nd.run_loop(chat, lambda c: executed.append(c) or "out", "task",
                      emit=emit, timeout_s=10_000)
    assert executed == ["first cmd"]                       # first only
    assert out["turns"] == 1
    asst = [r for r in rows if r.get("role") == "assistant"]
    assert asst[0]["protocol_violation"] == "multiple_tool_calls"
    assert len(asst[0]["tool_calls"]) == 2                 # transcript keeps all
    # Conversation history keeps ONLY the executed call (so every tool_call in
    # history has a matching tool reply); the tool reply pairs with c1.
    msgs = chat.calls[1]
    hist_asst = [m for m in msgs if m["role"] == "assistant"][0]
    assert [c["id"] for c in hist_asst["tool_calls"]] == ["c1"]
    assert [m["tool_call_id"] for m in msgs if m["role"] == "tool"] == ["c1"]


# --- invalid tool call: error reply, still a bounded turn --------------------

def test_invalid_args_reports_error_and_counts_turn():
    chat = FakeChat([
        _resp(tool_calls=[_tc("", call_id="cX", raw_args="{broken")]),
        _resp(content="done"),
    ])
    executed = []
    rows, emit = _collect_emit()
    out = nd.run_loop(chat, lambda c: executed.append(c) or "out", "task",
                      emit=emit, timeout_s=10_000)
    assert executed == []                                  # nothing executed
    assert out["ended"] == "done"
    assert out["turns"] == 1                               # still counts (bounded)
    trows = [r for r in rows if r.get("role") == "tool"]
    assert trows[0]["tool_call_id"] == "cX"
    assert trows[0]["content"].startswith("[driver] invalid tool call:")


# --- limits: max_turns and wall-clock timeout --------------------------------

def test_max_turns_ends_loop():
    chat = FakeChat([_resp(tool_calls=[_tc(f"echo {i}", call_id=f"c{i}")])
                     for i in range(5)])
    rows, emit = _collect_emit()
    out = nd.run_loop(chat, lambda c: "out", "task", emit=emit,
                      timeout_s=10_000, max_turns=3)
    assert out["ended"] == "max_turns"
    assert out["turns"] == 3
    assert len(chat.calls) == 3               # stopped exactly at the cap


def test_wall_clock_timeout_ends_loop():
    chat = FakeChat([_resp(tool_calls=[_tc("echo x")]),
                     _resp(tool_calls=[_tc("echo y")])])
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


# --- run_trial-side readback (trial_logic.parse_native_transcript) ----------

def test_parse_native_transcript_uses_result_total():
    rows, emit = _collect_emit()
    chat = FakeChat([
        _resp(tool_calls=[_tc("ls")],
              usage={"prompt_tokens": 100, "completion_tokens": 20}),
        _resp(content="summary",
              usage={"prompt_tokens": 130, "completion_tokens": 5}),
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
                    "tool_calls": [_tc("ls")],
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
