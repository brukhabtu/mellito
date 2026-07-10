"""Unit tests for infra/export_preferences.py — the preference-data exporter
(positives + explicit negatives). Pure functions + tmp_path fixture trees, so
no Modal/GPU/container/network/tokenizer is required — same style as
infra/tests/test_export_trajectories.py (whose fixture helpers are mirrored
here almost verbatim: one stream-json LINE per content block, shared
message.id spanning consecutive assistant lines, "(no content)" filtered by
the underlying transcript_to_messages/group_assistant_turns machinery).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export_preferences import (
    _diff_touched_paths,
    _percentile,
    build_preference_dataset,
    diff_touches_test_path,
    is_test_path,
)


# --- transcript fixture helpers (mirrors test_export_trajectories.py) ------

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


def _write_transcript(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(o) for o in lines) + "\n")


def _write_task(tasks_root, task, description):
    d = tasks_root / "dev" / task
    d.mkdir(parents=True, exist_ok=True)
    (d / "task.yaml").write_text(f"id: {task}\ndescription: |\n  {description}\n")


def _setup_run(tmp_path, run_id, variant, variant_claude_md, trial_specs):
    """Materialize one run dir (trials.jsonl + per-trial transcript/worker.diff)
    plus its variant's claude-config/CLAUDE.md, mirroring the real harness
    layout (experiments/runs/<run_id>/<task>/trial<N>/{transcript.jsonl,
    worker.diff} + summary.json{"variant": ...}).

    Each spec dict may set: task, trial, verdict, reason, thinking (str),
    filler (str, an extra assistant text block — used to control a
    trajectory's serialized length deterministically for cap-ordering tests),
    diff (worker.diff content), skip_transcript / skip_diff (bool, simulate a
    missing artifact).
    """
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    variant_dir = tmp_path / "variants" / variant
    (variant_dir / "claude-config").mkdir(parents=True, exist_ok=True)
    (variant_dir / "claude-config" / "CLAUDE.md").write_text(variant_claude_md)

    trials = []
    for spec in trial_specs:
        task, trial = spec["task"], spec["trial"]
        tdir = run_dir / task / f"trial{trial}"
        tdir.mkdir(parents=True, exist_ok=True)

        if not spec.get("skip_transcript"):
            mid = f"{task}-{trial}"
            lines = [
                _sys_line(),
                _asst(mid, {"type": "thinking", "thinking": spec.get("thinking", "")}),
                _asst(mid, {"type": "tool_use", "id": "t1", "name": "Bash",
                            "input": {"command": "pytest"}}),
                _tool_result("t1", "result"),
                _asst(mid, {"type": "text", "text": spec.get("filler", "")}),
                _result(),
            ]
            _write_transcript(tdir / "transcript.jsonl", lines)

        if not spec.get("skip_diff"):
            (tdir / "worker.diff").write_text(spec.get("diff", ""))

        default_reason = "verify_exit_zero" if spec["verdict"] == "pass" else "verify_exit_nonzero"
        trials.append({
            "task": task, "trial": trial, "verdict": spec["verdict"],
            "reason": spec.get("reason", default_reason),
            "transcript_path": f"{run_id}/{task}/trial{trial}/transcript.jsonl",
        })

    (run_dir / "trials.jsonl").write_text(
        "\n".join(json.dumps(t) for t in trials) + "\n")
    (run_dir / "summary.json").write_text(json.dumps({"variant": variant}))
    return run_dir


# --- test-path predicate ----------------------------------------------------

def test_is_test_path_predicate():
    true_cases = [
        "tests/test_foo.py",               # tests/ dir + basename, doubly true
        "package/tests/helpers.py",         # tests/ dir only (non-test basename)
        "testing/helpers.py",               # testing/ dir only (pytest's own layout)
        "test_foo.py",                      # root, basename only, no dir component
        "package/sub/test_foo.py",          # nested, basename match
        "conftest.py",                      # root, exact basename
        "package/conftest.py",              # nested, exact basename (no tests/ dir)
        "tests",                            # bare path component equal to "tests"
        "a/b/c/testing/d.py",               # testing/ dir deeply nested
    ]
    for p in true_cases:
        assert is_test_path(p) is True, p

    false_cases = [
        "src/module.py",                    # plain source
        "django/db/backends/postgresql/client.py",  # real corpus non-test path
        "verify_fix.py",                    # near-miss: not test_*/​*_test
        "classes_test.dot",                 # near-miss: right stem, wrong ext
        "contest.py",                       # near-miss: NOT literally conftest.py
        "attest_module.py",                 # contains "test" but no prefix/suffix match
        "reproduce_bug.py",
        "FIX_SUMMARY.md",
    ]
    for p in false_cases:
        assert is_test_path(p) is False, p


def test_diff_touched_paths_and_test_detection():
    all_source_diff = (
        "diff --git a/src/mod.py b/src/mod.py\n"
        "index abc..def 100644\n--- a/src/mod.py\n+++ b/src/mod.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/README.md b/README.md\n"
        "index abc..def 100644\n--- a/README.md\n+++ b/README.md\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    assert _diff_touched_paths(all_source_diff) == {"src/mod.py", "README.md"}
    assert diff_touches_test_path(all_source_diff) is False

    # rename INTO a tests/ directory must still be caught (via either endpoint).
    rename_diff = (
        "diff --git a/scripts/old_helper.py b/tests/new_helper.py\n"
        "similarity index 100%\n"
        "rename from scripts/old_helper.py\nrename to tests/new_helper.py\n"
    )
    assert _diff_touched_paths(rename_diff) == {
        "scripts/old_helper.py", "tests/new_helper.py"}
    assert diff_touches_test_path(rename_diff) is True

    # a brand-new conftest.py (basename-only match, no tests/ dir at all).
    new_conftest_diff = (
        "diff --git a/conftest.py b/conftest.py\n"
        "new file mode 100644\nindex 0000000..e69de29\n"
        "--- /dev/null\n+++ b/conftest.py\n@@ -0,0 +1 @@\n+import pytest\n"
    )
    assert diff_touches_test_path(new_conftest_diff) is True

    # empty diff (a real empty_diff fail's worker.diff) touches nothing.
    assert _diff_touched_paths("") == set()
    assert diff_touches_test_path("") is False


def test_percentile_linear_interpolation():
    vals = [10, 20, 30, 40]
    assert _percentile(vals, 0.0) == 10
    assert _percentile(vals, 1.0) == 40
    assert _percentile(vals, 0.5) == 25       # halfway between 20 and 30
    assert _percentile([], 0.5) == 0
    assert _percentile([7], 0.9) == 7


# --- end-to-end selection ----------------------------------------------------

def test_core_selection_rules(tmp_path):
    tasks_root = tmp_path / "tasks"
    _write_task(tasks_root, "taskA", "Fix the bug in taskA.")

    pos_id = "20260708T132147-v002-completion-contract"
    neg_id = "20260707T215242-v001-baseline"
    clean_diff = "diff --git a/src/mod.py b/src/mod.py\n"
    test_diff = "diff --git a/tests/test_mod.py b/tests/test_mod.py\n"

    v002_specs = [
        # trial0: kept positive — clean diff, has thinking.
        {"task": "taskA", "trial": 0, "verdict": "pass",
         "thinking": "clean reasoning", "diff": clean_diff},
        # trial1: excluded positive — diff touches a test path (thinking present,
        # so this pins that test_edit is what excludes it, not no_thinking).
        {"task": "taskA", "trial": 1, "verdict": "pass",
         "thinking": "reasoning here", "diff": test_diff},
        # trial2: excluded positive — no thinking at all (clean diff).
        {"task": "taskA", "trial": 2, "verdict": "pass",
         "thinking": "", "diff": clean_diff},
        # trial3: invalid — must be excluded entirely regardless of content.
        {"task": "taskA", "trial": 3, "verdict": "invalid",
         "reason": "worker_reported_error",
         "thinking": "would-be reasoning", "diff": clean_diff},
        # trial4: negative whose diff ALSO touches a test path — must be KEPT
        # (no test-edit screen for negatives).
        {"task": "taskA", "trial": 4, "verdict": "fail",
         "reason": "verify_exit_nonzero",
         "thinking": "reasoning for a fail", "diff": test_diff},
        # trial5: excluded positive — BOTH test-edit AND no-thinking; pins that
        # the test-edit screen is evaluated first (counted as test_edit, not
        # no_thinking).
        {"task": "taskA", "trial": 5, "verdict": "pass",
         "thinking": "", "diff": test_diff},
    ]
    v001_specs = [
        # trial0: negative with NO thinking at all — must be KEPT (has_thinking
        # gate is positives-only).
        {"task": "taskA", "trial": 0, "verdict": "fail", "reason": "empty_diff",
         "thinking": "", "diff": ""},
        # trial1: a PASS verdict in the non-positive run — never a candidate.
        {"task": "taskA", "trial": 1, "verdict": "pass",
         "thinking": "irrelevant", "diff": clean_diff},
    ]

    pos_run_dir = _setup_run(tmp_path, pos_id, "v002-completion-contract",
                             "V002 SYSTEM PROMPT", v002_specs)
    neg_run_dir = _setup_run(tmp_path, neg_id, "v001-baseline",
                             "V001 SYSTEM PROMPT", v001_specs)

    out = tmp_path / "pref.jsonl"
    stats = build_preference_dataset(
        pos_run=pos_run_dir, neg_runs=[neg_run_dir, pos_run_dir],
        tasks_root=tasks_root, variants_root=tmp_path / "variants",
        out_path=out, pos_cap=3, neg_cap=3,
    )

    rows = [json.loads(l) for l in out.read_text().splitlines()]
    by_key = {(r["run_id"], r["trial"], r["label"]): r for r in rows}

    positives = [r for r in rows if r["label"] == "pass"]
    assert len(positives) == 1
    assert positives[0]["run_id"] == pos_id
    assert positives[0]["trial"] == 0
    assert positives[0]["task"] == "taskA"
    assert positives[0]["messages"][0] == {"role": "system", "content": "V002 SYSTEM PROMPT"}

    # exact output-row schema (the trainer's contract) — no more, no fewer keys.
    assert set(positives[0].keys()) == {
        "task", "trial", "run_id", "label", "messages",
        "n_assistant_turns", "n_tool_calls", "has_thinking", "thinking_chars",
    }

    negatives = {(r["run_id"], r["trial"]) for r in rows if r["label"] == "fail"}
    assert negatives == {(pos_id, 4), (neg_id, 0)}

    neg_from_v001 = by_key[(neg_id, 0, "fail")]
    neg_from_v002 = by_key[(pos_id, 4, "fail")]
    assert neg_from_v001["messages"][0]["content"] == "V001 SYSTEM PROMPT"
    assert neg_from_v002["messages"][0]["content"] == "V002 SYSTEM PROMPT"
    assert neg_from_v001["has_thinking"] is False     # kept despite no thinking
    assert neg_from_v002["label"] == "fail"

    assert stats["totals_by_label"] == {"pass": 1, "fail": 2, "total": 3}
    assert stats["excluded"]["invalid"] == 1
    assert stats["excluded"]["test_edit"] == 2         # trial1 AND trial5
    assert stats["excluded"]["no_thinking"] == 1       # trial2 only (trial5 short-circuits on test_edit)
    assert stats["excluded"]["pass_ineligible_run"] == 1
    assert stats["excluded"]["missing_artifact"] == 0
    assert stats["excluded"]["cap"] == 0
    assert stats["kept_negative_reason_mix"] == {
        "verify_exit_nonzero": 1, "empty_diff": 1}

    prc = stats["per_run_counts"]
    assert prc[pos_id]["trials_total"] == 6
    assert prc[pos_id]["verdict_counts"] == {"pass": 4, "invalid": 1, "fail": 1}
    assert prc[pos_id]["kept_pass"] == 1
    assert prc[pos_id]["kept_fail"] == 1
    assert prc[neg_id]["trials_total"] == 2
    assert prc[neg_id]["verdict_counts"] == {"fail": 1, "pass": 1}
    assert prc[neg_id]["kept_pass"] == 0
    assert prc[neg_id]["kept_fail"] == 1

    hist = stats["length_histogram_chars_over_4"]
    assert hist["pass"]["n"] == 1
    assert hist["fail"]["n"] == 2
    for label in ("pass", "fail"):
        assert hist[label]["p50"] >= 0
        assert hist[label]["max"] >= hist[label]["p50"]


def test_per_task_caps_shortest_and_empty_diff_priority(tmp_path):
    tasks_root = tmp_path / "tasks"
    _write_task(tasks_root, "taskB", "Fix the bug in taskB.")

    v002_id = "20260708T132147-v002-completion-contract"
    v001_id = "20260707T215242-v001-baseline"
    clean_diff = "diff --git a/src/mod.py b/src/mod.py\n"

    v002_specs = [
        # four clean, thinking-bearing positives of strictly increasing length.
        {"task": "taskB", "trial": 0, "verdict": "pass", "thinking": "r",
         "diff": clean_diff, "filler": ""},
        {"task": "taskB", "trial": 1, "verdict": "pass", "thinking": "r",
         "diff": clean_diff, "filler": "x" * 50},
        {"task": "taskB", "trial": 2, "verdict": "pass", "thinking": "r",
         "diff": clean_diff, "filler": "x" * 200},
        {"task": "taskB", "trial": 3, "verdict": "pass", "thinking": "r",
         "diff": clean_diff, "filler": "x" * 500},
        # two same-priority-tier negatives (verify_exit_nonzero); trial5 shorter.
        {"task": "taskB", "trial": 4, "verdict": "fail",
         "reason": "verify_exit_nonzero", "diff": "", "filler": "x" * 100},
        {"task": "taskB", "trial": 5, "verdict": "fail",
         "reason": "verify_exit_nonzero", "diff": "", "filler": ""},
    ]
    v001_specs = [
        # the empty_diff negative: LONGEST content of all, but must still win
        # a slot first (reason priority beats length).
        {"task": "taskB", "trial": 0, "verdict": "fail", "reason": "empty_diff",
         "diff": "", "filler": "x" * 300},
        # a same-priority-tier (verify_exit_nonzero), mid-length competitor.
        {"task": "taskB", "trial": 1, "verdict": "fail",
         "reason": "verify_exit_nonzero", "diff": "", "filler": "x" * 50},
    ]

    v002_run_dir = _setup_run(tmp_path, v002_id, "v002-completion-contract", "SYS", v002_specs)
    v001_run_dir = _setup_run(tmp_path, v001_id, "v001-baseline", "SYS", v001_specs)

    out = tmp_path / "pref.jsonl"
    stats = build_preference_dataset(
        pos_run=v002_run_dir, neg_runs=[v001_run_dir, v002_run_dir],
        tasks_root=tasks_root, variants_root=tmp_path / "variants",
        out_path=out, pos_cap=2, neg_cap=2,
    )

    rows = [json.loads(l) for l in out.read_text().splitlines()]
    positives = sorted(r["trial"] for r in rows if r["label"] == "pass")
    assert positives == [0, 1]     # the two SHORTEST kept, longer two dropped

    negatives = {(r["run_id"], r["trial"]) for r in rows if r["label"] == "fail"}
    # empty_diff (v001 trial0) always wins a slot despite being the longest;
    # the other slot goes to the shortest same-priority-tier candidate
    # (v002 trial5), NOT the closer-by-run-but-longer v002 trial4 or v001 trial1.
    assert negatives == {(v001_id, 0), (v002_id, 5)}

    assert stats["excluded"]["cap"] == 4    # 2 dropped positives + 2 dropped negatives
    assert stats["kept_negative_reason_mix"] == {
        "empty_diff": 1, "verify_exit_nonzero": 1}


def test_missing_artifacts_are_skipped_and_logged(tmp_path):
    tasks_root = tmp_path / "tasks"
    _write_task(tasks_root, "taskC", "Fix the bug in taskC.")

    v002_id = "20260708T132147-v002-completion-contract"
    v001_id = "20260707T215242-v001-baseline"
    clean_diff = "diff --git a/src/mod.py b/src/mod.py\n"

    v002_specs = [
        # positive missing its worker.diff entirely.
        {"task": "taskC", "trial": 0, "verdict": "pass", "thinking": "r",
         "diff": clean_diff, "skip_diff": True},
        # positive missing its transcript entirely.
        {"task": "taskC", "trial": 1, "verdict": "pass", "thinking": "r",
         "diff": clean_diff, "skip_transcript": True},
        # one clean, fully-present positive so the task isn't entirely empty.
        {"task": "taskC", "trial": 2, "verdict": "pass", "thinking": "r",
         "diff": clean_diff},
    ]
    v001_specs = [
        # negative missing its worker.diff.
        {"task": "taskC", "trial": 0, "verdict": "fail",
         "reason": "verify_exit_nonzero", "diff": "", "skip_diff": True},
        # negative missing its transcript.
        {"task": "taskC", "trial": 1, "verdict": "fail",
         "reason": "verify_exit_nonzero", "diff": "", "skip_transcript": True},
    ]

    v002_run_dir = _setup_run(tmp_path, v002_id, "v002-completion-contract", "SYS", v002_specs)
    v001_run_dir = _setup_run(tmp_path, v001_id, "v001-baseline", "SYS", v001_specs)

    out = tmp_path / "pref.jsonl"
    stats = build_preference_dataset(
        pos_run=v002_run_dir, neg_runs=[v001_run_dir, v002_run_dir],
        tasks_root=tasks_root, variants_root=tmp_path / "variants",
        out_path=out, pos_cap=3, neg_cap=3,
    )

    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["trial"] == 2 and rows[0]["label"] == "pass"
    assert stats["excluded"]["missing_artifact"] == 4     # 2 positives + 2 negatives
