#!/usr/bin/env python3
"""Retrospective best-of-k SELECTION analyzer for a completed eval sweep.

WHY this module exists (P/B "selection-vs-single-shot" question): every
scaffold sweep runs 5 trials/task, but the harness ships a single-shot
verdict (majority-of-5). This module asks a different, retrospective
question: if the WORKER's own in-context Bash signal (did it run VERIFY and
did VERIFY look like it passed?) were used to pick exactly one trial per
task, how much of the pass@5 ceiling would a plausible, non-oracle selector
recover? It never looks at held-out data or changes any run; it only
re-reads existing trials.jsonl + transcripts + worker.diff, per
.claude/rules/experiment-integrity.md (numbers cite a run ID; nothing here
mutates a run).

THE FROZEN SELECTION RULE (pre-registered; implemented verbatim, not
"improved" on). Per task, over its (up to 5) trials, pick exactly ONE trial:

  TIER 1 -- trials where the worker EXECUTED the task's VERIFY command via a
  Bash tool call AND the LAST such execution in that trial returned a
  non-error tool_result (is_error == false). Lowest trial index wins among
  tier-1 trials.

  TIER 2 (no tier-1 trial exists for this task) -- trials whose worker.diff
  touches package source, i.e. is not scratch-only. Lowest index wins.

  TIER 3 (neither) -- the lowest-index VALID trial (valid = verdict !=
  "invalid"). This is the current single-shot behaviour and gives no
  selection benefit for that task.

A task counts as selected_solve iff the ORACLE verdict (trials.jsonl
`verdict` field -- never the worker's own belief) of the SELECTED trial is
"pass".

WORKER-VERIFY DETECTION -- how "the worker executed VERIFY" is operationalized
(documented here for audit, per the task spec's transparency requirement):

  A Bash tool_use `command` string counts as a VERIFY execution if ANY of:
    (a) SIGNATURE MATCH -- it contains the verify command's test-runner
        basename (e.g. "pytest", "runtests.py" -- derived by taking the last
        `&&`-separated segment of the task's `verify` string, since the
        earlier segments are just `source .../activate && cd /testbed`
        environment setup, and treating its first shell token as the
        runner) AND at least one "target token" -- a non-flag, non-numeric
        trailing argument from that same segment (e.g.
        "astropy/modeling/tests/test_separable.py" or
        "serializers.models.data"). This is the primary, high-precision
        detector and matches how every real trial in the corpus actually
        re-runs VERIFY (it re-types the exact command after reading it).
    (b) CAT/BASH/SH ON VERIFY.txt -- the command runs `cat`, `bash`, or `sh`
        against a path ending in VERIFY.txt (e.g. "cat VERIFY.txt", "cat
        /testbed/VERIFY.txt"). Per the pre-registered spec this counts even
        though a plain `cat` only *displays* the file rather than executing
        it -- a known, deliberately generous edge of the rule, called out
        here so an auditor can see exactly which trials it affects via the
        emitted `matched_command` field (grep the per-trial table for a
        `matched_command` that starts with "cat").
    (c) SAME/NEXT-TURN AFTER A VERIFY.txt READ -- the command invokes the
        same test-runner (by runner-basename substring only, no target-token
        requirement) in the same assistant turn as, or the assistant turn
        immediately following, a Read tool_use of a path ending in
        VERIFY.txt. This is a fallback for the case where the re-typed
        command's target arguments don't literally reappear (comment
        rewritten, extra pytest flags, etc.) but the causal link to VERIFY.txt
        is still visible in the transcript.
  Among all matched Bash commands in the trial, the LAST one (transcript
  order) determines the outcome: its paired tool_result's `is_error` decides
  RAN_PASS (is_error is False) vs RAN_FAIL (is_error is True or missing/None
  -- no positive signal, so we do not credit it). A trial with zero matches
  is NEVER_RAN. These are the only three states recorded.

  NOTE on why we cannot just reuse `group_assistant_turns`' tool turns for
  the is_error lookup: that function (infra/export_trajectories.py) is an
  SFT-prep helper and deliberately drops `is_error` from its `tool_result`
  turn dicts (it is training-irrelevant there). We still reuse it for
  canonical turn/assistant-id grouping (needed for the same/next-turn
  Read->Bash adjacency check), but is_error itself is recovered from the raw
  transcript lines directly via `_iter_objs` (also reused) into a
  tool_use_id -> is_error map, independent of that stripped structure.

SOURCE-EDIT DETECTION (tier 2), reusing the spirit of the P5 T6
"no-source-edit" definition from FINDINGS.md/v005-script-first's manifest: a
trial "touches package source" iff worker.diff contains >=1
`diff --git a/<path> b/<path>` line where <path> has a directory separator
(i.e. is not a workspace-root file) AND its basename does not look like a
throwaway scratch/repro script. Root-level files (no "/" in the git path --
this already covers root `test_*.py`, `FIX_SUMMARY.md`, `reproduce_bug.py`,
etc. uniformly) are always scratch. For paths that DO have a directory
separator, only `repro*.py` / `trace*.py` / `reproduce*.py` / `verify_*.py`
basenames are additionally treated as scratch -- deliberately NOT
`test_*.py`, because real repo test files legitimately live in nested
directories (e.g. `astropy/utils/tests/test_introspection.py`,
`sklearn/tests/test_base.py`) and must count as source edits; every such
diff observed in the actual runs also touches a real non-test source file
in the same diff, so this scoping choice does not change any tier-2 outcome
in the analyzed runs (verified by direct inspection during authoring).
"""
import argparse
import json
import re
import shlex
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
_INFRA_DIR = Path(__file__).resolve().parent
if str(_INFRA_DIR) not in sys.path:
    sys.path.insert(0, str(_INFRA_DIR))

from export_trajectories import _iter_objs, group_assistant_turns, _local_transcript_path  # noqa: E402
from sweep_stats import per_task as _sweep_per_task, _solved as _sweep_solved  # noqa: E402


# --- scratch-file / source-edit detection (TIER 2) ------------------------

_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+) b/.+$", re.MULTILINE)

_SCRATCH_BASENAME_PATTERNS = (
    re.compile(r"^repro.*\.py$", re.IGNORECASE),
    re.compile(r"^trace.*\.py$", re.IGNORECASE),
    re.compile(r"^reproduce.*\.py$", re.IGNORECASE),
    re.compile(r"^verify_.*\.py$", re.IGNORECASE),
)


def diff_paths(diff_text):
    """Return the `a/<path>` targets of every `diff --git` header in a
    worker.diff blob (empty list for an empty/missing diff)."""
    if not diff_text:
        return []
    return _DIFF_GIT_RE.findall(diff_text)


def is_scratch_path(path):
    """True if `path` is a workspace-root scratch file (see module docstring
    for the exact scoping of the basename patterns vs. the root check)."""
    if "/" not in path:
        return True
    basename = path.rsplit("/", 1)[-1]
    return any(pat.match(basename) for pat in _SCRATCH_BASENAME_PATTERNS)


def has_source_edit(diff_text):
    """True iff worker.diff touches at least one non-scratch package-source
    path (tier-2 predicate)."""
    return any(not is_scratch_path(p) for p in diff_paths(diff_text))


# --- VERIFY-command signature parsing --------------------------------------

def parse_verify_signature(verify_cmd):
    """Extract {"runner": basename, "targets": [tokens]} from a task's
    `verify` shell string. See module docstring (a)."""
    if not verify_cmd:
        return {"runner": "", "targets": []}
    segment = verify_cmd.split("&&")[-1].strip()
    try:
        tokens = shlex.split(segment)
    except ValueError:
        tokens = segment.split()
    if not tokens:
        return {"runner": "", "targets": []}
    runner = tokens[0].rsplit("/", 1)[-1]
    targets = [t for t in tokens[1:] if not t.startswith("-") and not t.isdigit()]
    return {"runner": runner, "targets": targets}


def _signature_match(cmd, sig):
    if not cmd or not sig.get("runner"):
        return False
    if sig["runner"] not in cmd:
        return False
    return any(target in cmd for target in sig["targets"])


_CAT_VERIFY_RE = re.compile(r"\b(?:cat|bash|sh)\b[^\n;&|]*VERIFY\.txt\b")


def _is_cat_verify_command(cmd):
    """See module docstring (b): cat/bash/sh applied to VERIFY.txt."""
    return bool(_CAT_VERIFY_RE.search(cmd or ""))


def _tool_result_error_map(lines):
    """tool_use_id -> is_error, read straight from the raw transcript (NOT
    via group_assistant_turns, which strips is_error -- see module
    docstring)."""
    out = {}
    for obj in _iter_objs(lines):
        if obj.get("type") != "user":
            continue
        msg = obj.get("message") or {}
        for b in msg.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                tid = b.get("tool_use_id")
                if tid is not None:
                    out[tid] = b.get("is_error")
    return out


def detect_worker_verify(lines, verify_cmd):
    """Return {"state": "RAN_PASS"|"RAN_FAIL"|"NEVER_RAN",
    "matched_command": str|None, "tool_use_id": str|None} for one trial's
    transcript lines, per the worker-VERIFY detection rule in the module
    docstring."""
    turns = group_assistant_turns(lines)
    error_map = _tool_result_error_map(lines)
    sig = parse_verify_signature(verify_cmd)

    asst_positions = [i for i, t in enumerate(turns) if t["role"] == "assistant"]
    read_verify_seq = set()
    for seq, i in enumerate(asst_positions):
        for b in turns[i]["content"]:
            if b.get("type") == "tool_use" and b.get("name") == "Read":
                fp = (b.get("input") or {}).get("file_path") or ""
                if fp.endswith("VERIFY.txt"):
                    read_verify_seq.add(seq)

    matches = []  # (seq, tool_use_id, command) in transcript order
    for seq, i in enumerate(asst_positions):
        after_verify_read = seq in read_verify_seq or (seq - 1) in read_verify_seq
        for b in turns[i]["content"]:
            if b.get("type") != "tool_use" or b.get("name") != "Bash":
                continue
            cmd = (b.get("input") or {}).get("command") or ""
            is_match = (
                _signature_match(cmd, sig)
                or _is_cat_verify_command(cmd)
                or (after_verify_read and bool(sig.get("runner")) and sig["runner"] in cmd)
            )
            if is_match:
                matches.append((seq, b.get("id"), cmd))

    if not matches:
        return {"state": "NEVER_RAN", "matched_command": None, "tool_use_id": None}

    _, tool_use_id, cmd = matches[-1]
    is_error = error_map.get(tool_use_id)
    state = "RAN_PASS" if is_error is False else "RAN_FAIL"
    return {"state": state, "matched_command": cmd, "tool_use_id": tool_use_id}


# --- per-task trial selection (the frozen tiered rule) ---------------------

def select_trial(task_trials):
    """task_trials: list of per-trial dicts, each with `trial` (int),
    `verdict` (str), `worker_verify_state` (str), `has_source_edit` (bool).
    Returns {"selected_trial": int|None, "tier": 1|2|3|None} -- lowest trial
    index wins within whichever tier fires first."""
    by_idx = sorted(task_trials, key=lambda t: t["trial"])

    tier1 = [t for t in by_idx if t["worker_verify_state"] == "RAN_PASS"]
    if tier1:
        return {"selected_trial": tier1[0]["trial"], "tier": 1}

    tier2 = [t for t in by_idx if t["has_source_edit"]]
    if tier2:
        return {"selected_trial": tier2[0]["trial"], "tier": 2}

    tier3 = [t for t in by_idx if t["verdict"] != "invalid"]
    if tier3:
        return {"selected_trial": tier3[0]["trial"], "tier": 3}

    return {"selected_trial": None, "tier": None}


# --- orchestration ----------------------------------------------------------

def _load_verify_cmd(tasks_root, task):
    for split in ("dev", "staging", "holdout"):
        p = Path(tasks_root) / split / task / "task.yaml"
        if p.exists():
            spec = yaml.safe_load(p.read_text()) or {}
            return spec.get("verify")
    return None


def analyze_run(run_dir, tasks_root=None):
    """The main entry point: read one run's trials.jsonl + transcripts +
    worker.diffs and return the full report dict described in the module
    docstring / task spec (selection metrics, per-task table, signal
    quality, confusion matrix)."""
    run_dir = Path(run_dir)
    tasks_root = Path(tasks_root) if tasks_root else ROOT / "tasks"

    trials_raw = list(_iter_objs(
        (run_dir / "trials.jsonl").read_text().splitlines()))

    by_task = defaultdict(list)
    for t in trials_raw:
        by_task[t["task"]].append(t)

    missing_task_yaml = []
    per_trial_rows = []
    per_task_table = []

    for task in sorted(by_task):
        trials = by_task[task]
        verify_cmd = _load_verify_cmd(tasks_root, task)
        if verify_cmd is None:
            missing_task_yaml.append(task)

        computed = []
        for t in trials:
            trial_idx = t.get("trial")
            tpath = _local_transcript_path(run_dir, task, trial_idx,
                                           t.get("transcript_path"))
            diff_path = run_dir / task / f"trial{trial_idx}" / "worker.diff"

            if verify_cmd is not None and tpath.exists():
                wv = detect_worker_verify(tpath.read_text().splitlines(),
                                          verify_cmd)
            else:
                wv = {"state": "NEVER_RAN", "matched_command": None,
                      "tool_use_id": None}

            diff_text = diff_path.read_text() if diff_path.exists() else ""
            row = {
                "task": task,
                "trial": trial_idx,
                "verdict": t.get("verdict"),
                "valid": t.get("verdict") != "invalid",
                "worker_verify_state": wv["state"],
                "matched_command": wv["matched_command"],
                "has_source_edit": has_source_edit(diff_text),
            }
            computed.append(row)
            per_trial_rows.append(row)

        sel = select_trial(computed)
        sel_idx = sel["selected_trial"]
        sel_row = None
        if sel_idx is not None:
            sel_row = next(c for c in computed if c["trial"] == sel_idx)

        per_task_table.append({
            "task": task,
            "selected_trial_idx": sel_idx,
            "tier": sel["tier"],
            "worker_verify_state": sel_row["worker_verify_state"] if sel_row else None,
            "oracle_verdict_of_selected": sel_row["verdict"] if sel_row else None,
            "passes": sum(1 for c in computed if c["verdict"] == "pass"),
            "valid": sum(1 for c in computed if c["valid"]),
        })

    n_tasks = len(per_task_table)
    selected_solve = sum(1 for r in per_task_table
                         if r["oracle_verdict_of_selected"] == "pass")

    # majority_solve / pass@5 / mean pass@1 -- reuse sweep_stats, the
    # existing single source of truth for "solved" (>=0.5 of valid trials).
    pt = _sweep_per_task(trials_raw)
    majority_solve = sum(1 for s in pt.values() if _sweep_solved(s))
    pass_at_5 = sum(1 for s in pt.values() if s["valid"] and s["passes"] >= 1)
    valid_trials_n = sum(1 for t in trials_raw if t.get("verdict") != "invalid")
    total_passes = sum(1 for t in trials_raw if t.get("verdict") == "pass")
    mean_pass_at_1 = (total_passes / valid_trials_n) if valid_trials_n else None

    # signal quality: P(oracle pass | worker state), at the TRIAL level.
    def _cond_pass_rate(state):
        rows = [r for r in per_trial_rows
                if r["worker_verify_state"] == state and r["valid"]]
        if not rows:
            return None
        return sum(1 for r in rows if r["verdict"] == "pass") / len(rows)

    signal_quality = {
        "P(oracle_pass|RAN_PASS)": _cond_pass_rate("RAN_PASS"),
        "P(oracle_pass|RAN_FAIL)": _cond_pass_rate("RAN_FAIL"),
        "P(oracle_pass|NEVER_RAN)": _cond_pass_rate("NEVER_RAN"),
        "n_RAN_PASS": sum(1 for r in per_trial_rows if r["worker_verify_state"] == "RAN_PASS"),
        "n_RAN_FAIL": sum(1 for r in per_trial_rows if r["worker_verify_state"] == "RAN_FAIL"),
        "n_NEVER_RAN": sum(1 for r in per_trial_rows if r["worker_verify_state"] == "NEVER_RAN"),
    }

    # 2x2 confusion matrix over trials that ran VERIFY (worker_verify_pass
    # vs oracle_pass); invalid-verdict trials count as "not pass".
    ran_rows = [r for r in per_trial_rows
               if r["worker_verify_state"] in ("RAN_PASS", "RAN_FAIL")]
    confusion_matrix = {
        "worker_pass_oracle_pass": sum(
            1 for r in ran_rows
            if r["worker_verify_state"] == "RAN_PASS" and r["verdict"] == "pass"),
        "worker_pass_oracle_not_pass": sum(
            1 for r in ran_rows
            if r["worker_verify_state"] == "RAN_PASS" and r["verdict"] != "pass"),
        "worker_fail_oracle_pass": sum(
            1 for r in ran_rows
            if r["worker_verify_state"] == "RAN_FAIL" and r["verdict"] == "pass"),
        "worker_fail_oracle_not_pass": sum(
            1 for r in ran_rows
            if r["worker_verify_state"] == "RAN_FAIL" and r["verdict"] != "pass"),
        "n": len(ran_rows),
    }

    return {
        "run_id": run_dir.name,
        "n_tasks": n_tasks,
        "selected_solve": selected_solve,
        "majority_solve": majority_solve,
        "pass_at_5": pass_at_5,
        "mean_pass_at_1": round(mean_pass_at_1, 4) if mean_pass_at_1 is not None else None,
        "per_task_table": per_task_table,
        "signal_quality": signal_quality,
        "confusion_matrix": confusion_matrix,
        "missing_task_yaml": missing_task_yaml,
    }


def format_report(report):
    lines = []
    lines.append(f"=== selection_analysis: {report['run_id']} ===")
    lines.append(f"selected_solve : {report['selected_solve']}/{report['n_tasks']}")
    lines.append(f"majority_solve : {report['majority_solve']}/{report['n_tasks']}")
    lines.append(f"pass@5         : {report['pass_at_5']}/{report['n_tasks']}")
    lines.append(f"mean pass@1    : {report['mean_pass_at_1']}")
    if report["missing_task_yaml"]:
        lines.append(f"WARNING missing task.yaml for: {report['missing_task_yaml']}")
    lines.append("")
    lines.append("Per-task selection:")
    lines.append(f"  {'task':40s} {'sel':>4s} {'tier':>4s} {'worker_verify':>14s} "
                 f"{'oracle':>8s} {'passes/valid':>12s}")
    for row in report["per_task_table"]:
        lines.append(
            f"  {row['task']:40s} {str(row['selected_trial_idx']):>4s} "
            f"{str(row['tier']):>4s} {str(row['worker_verify_state']):>14s} "
            f"{str(row['oracle_verdict_of_selected']):>8s} "
            f"{row['passes']}/{row['valid']:>10d}")
    lines.append("")
    lines.append("Signal quality (trial level):")
    for k, v in report["signal_quality"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Confusion matrix (trials that ran VERIFY; worker vs oracle):")
    for k, v in report["confusion_matrix"].items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Retrospective best-of-k selection analyzer.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", help="run_id under experiments/runs/, or a full path")
    ap.add_argument("--runs-root", default=str(ROOT / "experiments" / "runs"))
    ap.add_argument("--tasks-root", default=str(ROOT / "tasks"))
    args = ap.parse_args(argv)

    run_dir = Path(args.run_id)
    if not run_dir.exists():
        run_dir = Path(args.runs_root) / args.run_id
    if not run_dir.exists():
        sys.exit(f"run dir not found: {args.run_id}")

    report = analyze_run(run_dir, tasks_root=args.tasks_root)
    print(format_report(report))
    print()
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
