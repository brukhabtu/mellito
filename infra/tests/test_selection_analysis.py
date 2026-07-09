"""Unit tests for infra/selection_analysis.py -- the retrospective best-of-k
selection analyzer. Pure functions over inline fixtures (synthetic
stream-json transcripts, synthetic worker.diff text, synthetic trials.jsonl
rows), mirroring infra/tests/test_export_trajectories.py's style: no Modal,
GPU, network, or tokenizer required, and no real run data is a test
dependency (the two real runs are exercised separately by running the CLI,
not by these tests).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selection_analysis import (
    parse_verify_signature,
    detect_worker_verify,
    is_scratch_path,
    has_source_edit,
    diff_paths,
    select_trial,
    analyze_run,
)


# --- shared fixture helpers (mirrors test_export_trajectories.py) ----------

def _sys_line(model="ornith-35b"):
    return {"type": "system", "subtype": "init",
            "tools": ["Read", "Edit", "Bash"], "model": model}


def _asst(mid, block):
    """One assistant LINE carrying exactly ONE content block (as CC emits)."""
    return {"type": "assistant",
            "message": {"id": mid, "role": "assistant", "content": [block],
                        "usage": {"input_tokens": 1, "output_tokens": 1}}}


def _tool_result(tid, content, is_error=None):
    return {"type": "user",
            "message": {"role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tid,
                                     "content": content, "is_error": is_error}]}}


def _result():
    return {"type": "result", "subtype": "success", "num_turns": 1,
            "is_error": False}


def _bash(mid, tid, command):
    return _asst(mid, {"type": "tool_use", "id": tid, "name": "Bash",
                       "input": {"command": command}})


def _read(mid, tid, file_path):
    return _asst(mid, {"type": "tool_use", "id": tid, "name": "Read",
                       "input": {"file_path": file_path}})


VERIFY_CMD = ("source /opt/miniconda3/bin/activate testbed && cd /testbed && "
             "pytest -rA astropy/modeling/tests/test_separable.py")


# --- parse_verify_signature -------------------------------------------------

def test_parse_verify_signature_extracts_runner_and_targets():
    sig = parse_verify_signature(VERIFY_CMD)
    assert sig["runner"] == "pytest"
    assert sig["targets"] == ["astropy/modeling/tests/test_separable.py"]


def test_parse_verify_signature_multi_target_and_flags():
    verify = ("source x && cd /testbed && ./tests/runtests.py --verbosity 2 "
             "--settings=test_sqlite --parallel 1 serializers.models.data "
             "serializers.test_data")
    sig = parse_verify_signature(verify)
    assert sig["runner"] == "runtests.py"
    assert sig["targets"] == ["serializers.models.data", "serializers.test_data"]
    # flags and their numeric values are excluded
    assert "--parallel" not in sig["targets"]
    assert "1" not in sig["targets"]


# --- detect_worker_verify: RAN_PASS / RAN_FAIL / NEVER_RAN ------------------

def test_ran_pass_when_last_matching_execution_is_non_error():
    lines = [
        _sys_line(),
        _read("A", "t0", "/testbed/VERIFY.txt"),
        _tool_result("t0", "the verify command text"),
        _bash("B", "t1", "pytest -rA astropy/modeling/tests/test_separable.py"),
        _tool_result("t1", "5 passed", is_error=False),
        _result(),
    ]
    r = detect_worker_verify(lines, VERIFY_CMD)
    assert r["state"] == "RAN_PASS"
    assert "test_separable.py" in r["matched_command"]


def test_ran_fail_when_last_matching_execution_errors():
    lines = [
        _sys_line(),
        _bash("A", "t1", "pytest -rA astropy/modeling/tests/test_separable.py"),
        _tool_result("t1", "1 failed", is_error=True),
        _result(),
    ]
    r = detect_worker_verify(lines, VERIFY_CMD)
    assert r["state"] == "RAN_FAIL"


def test_never_ran_when_no_bash_matches_signature():
    lines = [
        _sys_line(),
        _bash("A", "t1", "ls /testbed"),
        _tool_result("t1", "astropy  setup.py", is_error=False),
        _asst("B", {"type": "text", "text": "Looks fine."}),
        _result(),
    ]
    r = detect_worker_verify(lines, VERIFY_CMD)
    assert r["state"] == "NEVER_RAN"
    assert r["matched_command"] is None


def test_last_matching_execution_wins_over_earlier_ones():
    # First run errors, second (later) run succeeds -> overall RAN_PASS.
    lines = [
        _sys_line(),
        _bash("A", "t1", "pytest -rA astropy/modeling/tests/test_separable.py"),
        _tool_result("t1", "1 failed", is_error=True),
        _bash("B", "t2", "pytest -rA astropy/modeling/tests/test_separable.py -x"),
        _tool_result("t2", "5 passed", is_error=False),
        _result(),
    ]
    r = detect_worker_verify(lines, VERIFY_CMD)
    assert r["state"] == "RAN_PASS"
    assert r["tool_use_id"] == "t2"

    # Reverse order: passes first, fails last -> overall RAN_FAIL.
    lines2 = [
        _sys_line(),
        _bash("A", "t1", "pytest -rA astropy/modeling/tests/test_separable.py"),
        _tool_result("t1", "5 passed", is_error=False),
        _bash("B", "t2", "pytest -rA astropy/modeling/tests/test_separable.py -x"),
        _tool_result("t2", "1 failed", is_error=True),
        _result(),
    ]
    r2 = detect_worker_verify(lines2, VERIFY_CMD)
    assert r2["state"] == "RAN_FAIL"
    assert r2["tool_use_id"] == "t2"


def test_cat_verify_txt_counts_as_execution():
    # A bare `cat VERIFY.txt` is a documented generous edge of the rule: it
    # counts as a match, and cat's own (trivially successful) tool_result
    # decides the outcome when it is the LAST match.
    lines = [
        _sys_line(),
        _bash("A", "t1", "cat VERIFY.txt"),
        _tool_result("t1", VERIFY_CMD, is_error=False),
        _result(),
    ]
    r = detect_worker_verify(lines, VERIFY_CMD)
    assert r["state"] == "RAN_PASS"
    assert r["matched_command"] == "cat VERIFY.txt"


def test_same_next_turn_after_verify_read_counts_even_without_target_token():
    # The command that follows a Read of VERIFY.txt invokes the same runner
    # but with a rewritten/partial target list (no literal target-token
    # match) -- rule (c) still credits it.
    lines = [
        _sys_line(),
        _read("A", "r1", "/testbed/VERIFY.txt"),
        _tool_result("r1", VERIFY_CMD, is_error=False),
        _bash("B", "t1", "pytest -rA -k separable"),  # no literal target token
        _tool_result("t1", "5 passed", is_error=False),
        _result(),
    ]
    r = detect_worker_verify(lines, VERIFY_CMD)
    assert r["state"] == "RAN_PASS"
    assert r["tool_use_id"] == "t1"


def test_bash_two_turns_after_verify_read_does_not_count():
    # Adjacency is same-or-NEXT assistant turn only; a runner-only match two
    # turns after the Read must not be credited (would otherwise NEVER_RAN
    # here since there is no signature/cat match either).
    lines = [
        _sys_line(),
        _read("A", "r1", "/testbed/VERIFY.txt"),
        _tool_result("r1", VERIFY_CMD, is_error=False),
        _asst("B", {"type": "text", "text": "let me check something else first"}),
        _bash("C", "t1", "pytest -k unrelated"),
        _tool_result("t1", "ok", is_error=False),
        _result(),
    ]
    r = detect_worker_verify(lines, VERIFY_CMD)
    assert r["state"] == "NEVER_RAN"


# --- scratch-file exclusion / has_source_edit (TIER 2) ----------------------

def test_root_level_files_are_always_scratch():
    assert is_scratch_path("reproduce_bug.py") is True
    assert is_scratch_path("test_bug.py") is True       # root test_*.py
    assert is_scratch_path("FIX_SUMMARY.md") is True    # non-.py root file too
    assert is_scratch_path("trace_it.py") is True


def test_nested_scratch_basenames_are_scratch():
    assert is_scratch_path("some/dir/repro_issue.py") is True
    assert is_scratch_path("some/dir/verify_fix.py") is True
    assert is_scratch_path("some/dir/traceit.py") is True


def test_nested_real_test_file_is_not_scratch():
    # Real repo test files legitimately live in nested test directories and
    # must still count as package source (deliberately NOT scratch, unlike
    # a root-level test_*.py -- see module docstring).
    assert is_scratch_path("astropy/utils/tests/test_introspection.py") is False
    assert is_scratch_path("sklearn/tests/test_base.py") is False


def test_nested_real_source_file_is_not_scratch():
    assert is_scratch_path("astropy/modeling/separable.py") is False


def test_has_source_edit_true_when_any_non_scratch_path_present():
    diff = (
        "diff --git a/reproduce_bug.py b/reproduce_bug.py\n"
        "index 111..222 100644\n"
        "--- a/reproduce_bug.py\n"
        "+++ b/reproduce_bug.py\n"
        "@@ -1 +1 @@\n"
        "-old\n+new\n"
        "diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py\n"
        "index 333..444 100644\n"
        "--- a/astropy/modeling/separable.py\n"
        "+++ b/astropy/modeling/separable.py\n"
        "@@ -1 +1 @@\n"
        "-old\n+new\n"
    )
    assert diff_paths(diff) == ["reproduce_bug.py", "astropy/modeling/separable.py"]
    assert has_source_edit(diff) is True


def test_has_source_edit_false_when_only_scratch_files_touched():
    diff = (
        "diff --git a/reproduce_bug.py b/reproduce_bug.py\n"
        "index 111..222 100644\n"
        "--- a/reproduce_bug.py\n+++ b/reproduce_bug.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/test_bug.py b/test_bug.py\n"
        "index 555..666 100644\n"
        "--- a/test_bug.py\n+++ b/test_bug.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    assert has_source_edit(diff) is False


def test_has_source_edit_false_for_empty_diff():
    assert has_source_edit("") is False
    assert has_source_edit(None) is False


# --- select_trial: tier precedence + lowest-index tie-break -----------------

def _trial(idx, verdict, worker_verify_state="NEVER_RAN", has_source_edit=False):
    return {"trial": idx, "verdict": verdict,
            "worker_verify_state": worker_verify_state,
            "has_source_edit": has_source_edit}


def test_tier1_wins_over_tier2_and_tier3_and_picks_lowest_index():
    trials = [
        _trial(0, "fail", worker_verify_state="NEVER_RAN", has_source_edit=True),  # tier2 candidate
        _trial(1, "pass", worker_verify_state="RAN_PASS"),                         # tier1 candidate
        _trial(2, "pass", worker_verify_state="RAN_PASS"),                         # also tier1, higher idx
        _trial(3, "fail", worker_verify_state="NEVER_RAN"),
    ]
    sel = select_trial(trials)
    assert sel == {"selected_trial": 1, "tier": 1}


def test_tier2_used_only_when_no_tier1_and_picks_lowest_index():
    trials = [
        _trial(0, "fail", worker_verify_state="NEVER_RAN", has_source_edit=False),
        _trial(1, "fail", worker_verify_state="RAN_FAIL", has_source_edit=True),   # first source-edit trial
        _trial(2, "pass", worker_verify_state="NEVER_RAN", has_source_edit=True),  # later source-edit trial
    ]
    sel = select_trial(trials)
    assert sel == {"selected_trial": 1, "tier": 2}


def test_tier3_fallback_when_no_tier1_or_tier2_picks_lowest_valid_index():
    trials = [
        _trial(0, "invalid"),                     # not valid -- skipped
        _trial(1, "fail"),                        # lowest valid
        _trial(2, "pass"),
    ]
    sel = select_trial(trials)
    assert sel == {"selected_trial": 1, "tier": 3}


def test_select_trial_none_when_all_invalid_and_no_edits():
    trials = [_trial(0, "invalid"), _trial(1, "invalid")]
    sel = select_trial(trials)
    assert sel == {"selected_trial": None, "tier": None}


# --- analyze_run: end-to-end selected_solve counting ------------------------

def _write_transcript(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(o) for o in lines) + "\n")


def test_analyze_run_selected_solve_end_to_end(tmp_path):
    run_id = "20260101T000000-vtest"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    tasks_root = tmp_path / "tasks"

    verify = "source x && cd /testbed && pytest -rA pkg/tests/test_thing.py"

    # taskA: trial0 empty_diff/never-ran, trial1 worker RAN_PASS and oracle
    # pass -> tier1 selects trial1 -> selected_solve contributes 1.
    (tasks_root / "dev" / "taskA").mkdir(parents=True)
    (tasks_root / "dev" / "taskA" / "task.yaml").write_text(
        f"id: taskA\nverify: \"{verify}\"\n")

    _write_transcript(run_dir / "taskA" / "trial0" / "transcript.jsonl", [
        _sys_line(),
        _asst("A", {"type": "text", "text": "thinking, no bash run"}),
        _result(),
    ])
    (run_dir / "taskA" / "trial0").mkdir(parents=True, exist_ok=True)
    (run_dir / "taskA" / "trial0" / "worker.diff").write_text("")

    _write_transcript(run_dir / "taskA" / "trial1" / "transcript.jsonl", [
        _sys_line(),
        _bash("B", "t1", "pytest -rA pkg/tests/test_thing.py"),
        _tool_result("t1", "1 passed", is_error=False),
        _result(),
    ])
    (run_dir / "taskA" / "trial1").mkdir(parents=True, exist_ok=True)
    (run_dir / "taskA" / "trial1" / "worker.diff").write_text(
        "diff --git a/pkg/thing.py b/pkg/thing.py\n"
        "index 1..2 100644\n--- a/pkg/thing.py\n+++ b/pkg/thing.py\n"
        "@@ -1 +1 @@\n-old\n+new\n")

    # taskB: no trial ever runs VERIFY; trial0 has a scratch-only diff,
    # trial1 has a real source edit but oracle verdict is "fail" -> tier2
    # selects trial1, but selected_solve does NOT count it (oracle != pass).
    (tasks_root / "dev" / "taskB").mkdir(parents=True)
    (tasks_root / "dev" / "taskB" / "task.yaml").write_text(
        f"id: taskB\nverify: \"{verify}\"\n")

    _write_transcript(run_dir / "taskB" / "trial0" / "transcript.jsonl", [
        _sys_line(), _asst("A", {"type": "text", "text": "no bash at all"}), _result(),
    ])
    (run_dir / "taskB" / "trial0").mkdir(parents=True, exist_ok=True)
    (run_dir / "taskB" / "trial0" / "worker.diff").write_text(
        "diff --git a/reproduce_bug.py b/reproduce_bug.py\n"
        "index 1..2 100644\n--- a/reproduce_bug.py\n+++ b/reproduce_bug.py\n"
        "@@ -1 +1 @@\n-old\n+new\n")

    _write_transcript(run_dir / "taskB" / "trial1" / "transcript.jsonl", [
        _sys_line(), _asst("A", {"type": "text", "text": "no bash at all"}), _result(),
    ])
    (run_dir / "taskB" / "trial1").mkdir(parents=True, exist_ok=True)
    (run_dir / "taskB" / "trial1" / "worker.diff").write_text(
        "diff --git a/pkg/thing.py b/pkg/thing.py\n"
        "index 1..2 100644\n--- a/pkg/thing.py\n+++ b/pkg/thing.py\n"
        "@@ -1 +1 @@\n-old\n+new\n")

    trials = [
        {"task": "taskA", "trial": 0, "verdict": "fail",
         "transcript_path": f"{run_id}/taskA/trial0/transcript.jsonl"},
        {"task": "taskA", "trial": 1, "verdict": "pass",
         "transcript_path": f"{run_id}/taskA/trial1/transcript.jsonl"},
        {"task": "taskB", "trial": 0, "verdict": "fail",
         "transcript_path": f"{run_id}/taskB/trial0/transcript.jsonl"},
        {"task": "taskB", "trial": 1, "verdict": "fail",
         "transcript_path": f"{run_id}/taskB/trial1/transcript.jsonl"},
    ]
    (run_dir / "trials.jsonl").write_text(
        "\n".join(json.dumps(t) for t in trials) + "\n")

    report = analyze_run(run_dir, tasks_root=tasks_root)

    assert report["n_tasks"] == 2
    assert report["selected_solve"] == 1  # only taskA

    by_task = {r["task"]: r for r in report["per_task_table"]}
    assert by_task["taskA"]["selected_trial_idx"] == 1
    assert by_task["taskA"]["tier"] == 1
    assert by_task["taskA"]["oracle_verdict_of_selected"] == "pass"

    assert by_task["taskB"]["selected_trial_idx"] == 1
    assert by_task["taskB"]["tier"] == 2
    assert by_task["taskB"]["oracle_verdict_of_selected"] == "fail"

    # signal quality / confusion matrix sanity: taskA/trial1 is the only
    # RAN_PASS trial, and its oracle verdict is "pass".
    assert report["signal_quality"]["n_RAN_PASS"] == 1
    assert report["signal_quality"]["P(oracle_pass|RAN_PASS)"] == 1.0
    assert report["confusion_matrix"]["worker_pass_oracle_pass"] == 1
    assert report["confusion_matrix"]["worker_pass_oracle_not_pass"] == 0
