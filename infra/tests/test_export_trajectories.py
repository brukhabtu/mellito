"""Unit tests for infra/export_trajectories.py — the stream-json -> SFT
conversion. Pure functions over inline fixtures (lists of dicts), so no Modal,
GPU, container, or on-disk transcript is required. Fixtures mirror the real
ornith stream-json shapes verified against a live run (one content block per
line, same message.id spanning consecutive assistant lines, arbitrary block
order within a turn, empty "(no content)" text placeholders, chatcmpl-tool-…
ids), but are entirely self-contained — the one real pulled transcript is
gitignored/ephemeral and is never a test dependency.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export_trajectories import (
    group_assistant_turns,
    transcript_to_messages,
    build_dataset,
)


def _sys_line(model="ornith-35b"):
    return {"type": "system", "subtype": "init",
            "tools": ["Read", "Edit", "Bash"], "model": model}


def _asst(mid, block):
    """One assistant LINE carrying exactly ONE content block (as CC emits)."""
    return {"type": "assistant",
            "message": {"id": mid, "role": "assistant", "content": [block],
                        "usage": {"input_tokens": 1, "output_tokens": 1}}}


def _tool_result(tid, content):
    return {"type": "user",
            "message": {"role": "user",
                        "content": [{"type": "tool_result",
                                     "tool_use_id": tid, "content": content}]}}


def _result():
    return {"type": "result", "subtype": "success", "num_turns": 1,
            "is_error": False}


# (a) same-id blocks merge into ONE turn in canonical order; tool_results pair.
def test_grouping_merges_same_id_canonical_order_and_pairs_results():
    # Emitted order is text, then thinking, then tool_use — NOT canonical.
    lines = [
        _sys_line(),
        _asst("A", {"type": "text", "text": "let me look"}),
        _asst("A", {"type": "thinking", "thinking": "reasoning here",
                    "signature": "sig-A"}),
        _asst("A", {"type": "tool_use", "id": "chatcmpl-tool-1",
                    "name": "Read", "input": {"file_path": "/x.py"}}),
        _tool_result("chatcmpl-tool-1", "file body"),
        _asst("B", {"type": "text", "text": "done"}),
        _result(),
    ]
    turns = group_assistant_turns(lines)
    # turn A (assistant), tool result, turn B (assistant)
    assert [t["role"] for t in turns] == ["assistant", "tool", "assistant"]

    a = turns[0]
    assert a["id"] == "A"
    # canonical order: thinking, then text, then tool_use
    assert [b["type"] for b in a["content"]] == ["thinking", "text", "tool_use"]

    tool = turns[1]
    assert tool["content"][0]["tool_use_id"] == "chatcmpl-tool-1"
    assert tool["content"][0]["content"] == "file body"


# (b) thinking text preserved, signature dropped.
def test_thinking_preserved_signature_dropped():
    lines = [
        _sys_line(),
        _asst("A", {"type": "thinking", "thinking": "keep me",
                    "signature": "DROP"}),
        _result(),
    ]
    turns = group_assistant_turns(lines)
    tb = turns[0]["content"][0]
    assert tb["type"] == "thinking"
    assert tb["thinking"] == "keep me"
    assert "signature" not in tb


# (c) a transcript with zero thinking -> has_thinking False, thinking_chars 0.
def test_no_thinking_flags_false():
    lines = [
        _sys_line(),
        _asst("A", {"type": "text", "text": "just text"}),
        _asst("A", {"type": "tool_use", "id": "t1", "name": "Bash",
                    "input": {"command": "ls"}}),
        _tool_result("t1", "out"),
        _result(),
    ]
    ex = transcript_to_messages(lines, "SYS", "USER")
    assert ex["has_thinking"] is False
    assert ex["thinking_chars"] == 0
    assert ex["n_tool_calls"] == 1
    assert ex["n_assistant_turns"] == 1


# (d) tool_use / tool_result correlate by id (format-agnostic ids).
def test_tool_use_result_correlation_by_id():
    lines = [
        _sys_line(),
        _asst("A", {"type": "tool_use", "id": "toolu_abc", "name": "Read",
                    "input": {}}),
        _asst("A", {"type": "tool_use", "id": "toolu_def", "name": "Read",
                    "input": {}}),
        _tool_result("toolu_abc", "first"),
        _tool_result("toolu_def", "second"),
        _result(),
    ]
    ex = transcript_to_messages(lines, "SYS", "USER")
    # messages: system, user, assistant(2 tool_use), tool, tool
    assert ex["messages"][0]["role"] == "system"
    assert ex["messages"][1]["role"] == "user"
    asst = ex["messages"][2]
    ids = [b["id"] for b in asst["content"] if b["type"] == "tool_use"]
    assert ids == ["toolu_abc", "toolu_def"]
    tool_msgs = [m for m in ex["messages"] if m["role"] == "tool"]
    assert [m["content"][0]["tool_use_id"] for m in tool_msgs] == [
        "toolu_abc", "toolu_def"]
    assert ex["n_tool_calls"] == 2


# (e) empty / garbled lines are tolerated (mix of str and dict input).
def test_garbled_lines_tolerated():
    lines = [
        "",                                   # blank
        "{not valid json",                    # partial
        json.dumps(_sys_line()),              # raw JSON string
        "null",                               # valid JSON but not a dict
        _asst("A", {"type": "thinking", "thinking": "ok"}),  # dict
        json.dumps(_result()),
    ]
    ex = transcript_to_messages(lines, "SYS", "USER")
    assert ex["n_assistant_turns"] == 1
    assert ex["has_thinking"] is True


# "(no content)" placeholder text is filtered out of assistant targets.
def test_placeholder_text_dropped():
    lines = [
        _sys_line(),
        _asst("A", {"type": "text", "text": "(no content)"}),
        _asst("A", {"type": "thinking", "thinking": "real reasoning"}),
        _asst("A", {"type": "text", "text": "real answer"}),
        _result(),
    ]
    turns = group_assistant_turns(lines)
    types = [b["type"] for b in turns[0]["content"]]
    texts = [b["text"] for b in turns[0]["content"] if b["type"] == "text"]
    assert types == ["thinking", "text"]  # placeholder text removed
    assert texts == ["real answer"]


def _write_transcript(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(o) for o in lines) + "\n")


# build_dataset end-to-end: filters to passing, drops no-thinking, caps,
# writes JSONL, returns stats. Integrity: a failing trial never lands.
def test_build_dataset_filters_and_caps(tmp_path):
    run_id = "20260101T000000-vtest"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)

    variant_dir = tmp_path / "variants" / "vtest"
    (variant_dir / "claude-config").mkdir(parents=True)
    (variant_dir / "claude-config" / "CLAUDE.md").write_text("WORKER SYS PROMPT")

    tasks_root = tmp_path / "tasks"
    (tasks_root / "dev" / "taskX").mkdir(parents=True)
    (tasks_root / "dev" / "taskX" / "task.yaml").write_text(
        "id: taskX\ndescription: |\n  Fix the bug in taskX.\n")

    def good(mid, think, tid):
        return [
            _sys_line(),
            _asst(mid, {"type": "thinking", "thinking": think,
                        "signature": ""}),
            _asst(mid, {"type": "tool_use", "id": tid, "name": "Bash",
                        "input": {"command": "pytest"}}),
            _tool_result(tid, "ok"),
            _result(),
        ]

    # Two PASSING trials with distinct thinking (differ in length), one PASSING
    # trial with NO thinking (must be dropped), one FAILING trial (must never
    # appear regardless of content).
    _write_transcript(run_dir / "taskX" / "trial0" / "transcript.jsonl",
                      good("A", "short reasoning", "t0"))
    _write_transcript(run_dir / "taskX" / "trial1" / "transcript.jsonl",
                      good("B", "a much longer reasoning trace here", "t1"))
    _write_transcript(run_dir / "taskX" / "trial2" / "transcript.jsonl",
                      [_sys_line(),
                       _asst("C", {"type": "text", "text": "no thinking"}),
                       _result()])
    _write_transcript(run_dir / "taskX" / "trial3" / "transcript.jsonl",
                      good("D", "failing trial reasoning", "t3"))

    trials = [
        {"task": "taskX", "trial": 0, "verdict": "pass",
         "transcript_path": f"{run_id}/taskX/trial0/transcript.jsonl"},
        {"task": "taskX", "trial": 1, "verdict": "pass",
         "transcript_path": f"{run_id}/taskX/trial1/transcript.jsonl"},
        {"task": "taskX", "trial": 2, "verdict": "pass",
         "transcript_path": f"{run_id}/taskX/trial2/transcript.jsonl"},
        {"task": "taskX", "trial": 3, "verdict": "fail",
         "transcript_path": f"{run_id}/taskX/trial3/transcript.jsonl"},
    ]
    (run_dir / "trials.jsonl").write_text(
        "\n".join(json.dumps(t) for t in trials) + "\n")

    out = tmp_path / "sft.jsonl"
    stats = build_dataset(run_dir, variant_dir, tasks_root, out,
                          cap_per_task=3, min_thinking_chars=1)

    assert stats["n_passing_trials"] == 3  # the fail is excluded up front
    assert stats["n_examples"] == 2        # two passing-with-thinking kept
    assert stats["dropped_no_thinking"] == 1
    assert stats["n_tasks_covered"] == 1
    assert stats["per_task_counts"]["taskX"] == 2

    examples = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(examples) == 2
    # No failing trial leaked in.
    assert all(e["trial"] in (0, 1) for e in examples)
    # System = worker CLAUDE.md; user = task description.
    e = examples[0]
    assert e["messages"][0] == {"role": "system", "content": "WORKER SYS PROMPT"}
    assert "Fix the bug in taskX." in e["messages"][1]["content"]
    # Shortest-first cap ordering: trial0 (short reasoning) sorts before trial1.
    assert examples[0]["trial"] == 0


def test_build_dataset_cap_and_dedupe(tmp_path):
    run_id = "20260101T000000-vcap"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    variant_dir = tmp_path / "variants" / "vcap"
    (variant_dir / "claude-config").mkdir(parents=True)
    (variant_dir / "claude-config" / "CLAUDE.md").write_text("SYS")
    tasks_root = tmp_path / "tasks"
    (tasks_root / "dev" / "taskY").mkdir(parents=True)
    (tasks_root / "dev" / "taskY" / "task.yaml").write_text(
        "description: do the thing\n")

    def traj(mid, think, tid):
        return [_sys_line(),
                _asst(mid, {"type": "thinking", "thinking": think}),
                _asst(mid, {"type": "tool_use", "id": tid, "name": "Bash",
                            "input": {"command": "x"}}),
                _tool_result(tid, "ok"), _result()]

    # trial0 and trial1 are IDENTICAL model turns (same content) -> one is a
    # dedupe drop; trial2 is distinct. cap=1 then keeps only the single
    # shortest survivor.
    _write_transcript(run_dir / "taskY" / "trial0" / "transcript.jsonl",
                      traj("A", "same", "tid"))
    _write_transcript(run_dir / "taskY" / "trial1" / "transcript.jsonl",
                      traj("A", "same", "tid"))
    _write_transcript(run_dir / "taskY" / "trial2" / "transcript.jsonl",
                      traj("B", "different and much longer reasoning", "tid2"))
    trials = [
        {"task": "taskY", "trial": i, "verdict": "pass",
         "transcript_path": f"{run_id}/taskY/trial{i}/transcript.jsonl"}
        for i in range(3)]
    (run_dir / "trials.jsonl").write_text(
        "\n".join(json.dumps(t) for t in trials) + "\n")

    out = tmp_path / "sft.jsonl"
    stats = build_dataset(run_dir, variant_dir, tasks_root, out,
                          cap_per_task=1, min_thinking_chars=1)
    assert stats["n_examples"] == 1
    assert stats["dropped_by_cap"] >= 1  # dedupe + cap both counted here
