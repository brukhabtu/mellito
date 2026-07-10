#!/usr/bin/env python3
"""Build a preference-training JSONL (positives + explicit negatives) from
Claude Code `stream-json` transcripts already on disk under experiments/runs/.

WHY this module exists (P7 offline preference-tuning rung — see PLAN.md P4/P7,
findings/FINDINGS.md 2026-07-10 P7-C0): `export_trajectories.py` builds an SFT
(imitation) set from PASSING trajectories only. This module builds the richer
preference set the P7 recipe needs: PASSING trajectories as positives *and*
FAILING trajectories as explicit negatives, so a preference objective (KTO /
weighted-CE unlikelihood) can push down on the under-action failure mode that
imitation alone cannot touch (a failing trial is never a valid imitation
target, but it is exactly the shape of thing a negative-weighted loss wants).

Integrity (per .claude/rules/experiment-integrity.md, amendment 2026-07-10):
  - Failed trajectories never enter training data as imitation targets, but
    MAY serve as explicitly-negative examples in preference-based training.
  - POSITIVE candidates must be verifier-PASS **and** test-edit-clean (the
    worker's diff touches no test file) — a passing trial that only got there
    by editing the tests is not a positive no matter what the verdict says.
  - Negatives get NO test-edit screen: a fail is undesirable regardless of
    what it touched (a "gamed" fail is still not something to imitate
    negatively against; it just isn't a positive-candidate concern here).
  - `invalid` trials (execution error, not a real model outcome — "error !=
    fail" per PLAN.md) enter NOTHING: no positive, no negative.
  - Every reported number in the printed stats traces back to a run_id under
    experiments/runs/; holdout is never read (only tasks/dev/<task>/task.yaml
    is consulted for user-turn text — dev is the only split either run drew
    tasks from).

Positives are restricted to ONE run by design (see PLAN.md / FINDINGS
2026-07-08): the v001-baseline run predates the reasoning-capture fix, so its
transcripts carry no `<think>` content — using it for positives would silently
train against thinking-free targets. Negatives carry no such constraint (a
fail's transcript is evidence of what NOT to do regardless of whether it
happens to contain thinking), so both runs contribute negatives.

Reused, by import, from export_trajectories.py (this repo's SFT exporter) —
deliberately NOT reimplemented, so both exporters agree bit-for-bit on what a
"trajectory" and a "message list" look like:
  - `_iter_objs`            — tolerant per-line JSON parsing.
  - `_local_transcript_path`— trials.jsonl transcript_path -> local file path.
  - `transcript_to_messages`— the stream-json -> Anthropic-shaped messages
    conversion (system/user/assistant/tool turns; thinking preserved,
    signature dropped; canonical block order).
See that module's docstring for the shape facts driving those functions.
"""

import argparse
import fnmatch
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

# --- import the SFT exporter's reusable primitives ---------------------
# export_trajectories.py lives directly beside this file in infra/, so this
# insert is a (harmless, one-`dirname`) belt-and-braces echo of the pattern
# used by infra/tests/test_export_trajectories.py — that test file sits one
# directory deeper (infra/tests/), so it needs TWO `dirname()` calls to reach
# infra/; this file already lives IN infra/, so ONE call reaches it. Either
# way the target directory placed on sys.path is the same: infra/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_trajectories import (  # noqa: E402  (import after sys.path fix-up)
    _iter_objs,
    _local_transcript_path,
    transcript_to_messages,
)

ROOT = Path(__file__).resolve().parent.parent

# Hardcoded default runs (overridable via --pos-run / --neg-run). Per the
# output contract: POSITIVES may only ever come from the recapture run
# (thinking-preserving); NEGATIVES are drawn from both.
DEFAULT_POS_RUN = ROOT / "experiments" / "runs" / "20260708T132147-v002-completion-contract"
DEFAULT_NEG_RUNS = [
    ROOT / "experiments" / "runs" / "20260707T215242-v001-baseline",
    ROOT / "experiments" / "runs" / "20260708T132147-v002-completion-contract",
]

# len(text) / this ~= token count. Same order-of-magnitude proxy convention as
# export_trajectories.py's CHARS_PER_TOKEN — never cite as an exact count.
CHARS_PER_TOKEN = 4.0

# A unified git diff header line always looks like `diff --git a/X b/Y`
# regardless of add/delete/rename (renames just have X != Y; adds/deletes have
# X == Y with the content living under /dev/null on one side) — so parsing
# only this line is sufficient to recover every path the diff touches.
# Assumes paths contain no whitespace (true of every worker.diff in this
# corpus, verified empirically); a path WITH whitespace would simply fail to
# match here (fails closed: that file is never flagged as a test path, never
# silently mis-parsed).
_DIFF_GIT_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)\s*$")


# --- test-path predicate --------------------------------------------------

def _diff_touched_paths(diff_text):
    """Return the set of file paths named in a git-diff blob's headers."""
    paths = set()
    for line in diff_text.splitlines():
        m = _DIFF_GIT_RE.match(line)
        if m:
            paths.add(m.group(1))
            paths.add(m.group(2))
    return paths


def is_test_path(path):
    """True if `path` (a POSIX-style relative path string) is a TEST path.

    Exact rule (per spec, implemented literally, no extra leniency):
      - the basename matches `test_*.py` or `*_test.py` or is `conftest.py`,
        OR
      - ANY path component (directory OR basename) equals "tests" or
        "testing".
    Both conditions are checked independently (OR): a non-test-named file
    living inside a `tests/`/`testing/` directory is still a TEST path (e.g.
    `sklearn/ensemble/tests/__init__.py`), and a test-named file at repo root
    with no directory at all is still a TEST path (e.g. `test_repro.py`).
    """
    parts = path.split("/")
    if any(part in ("tests", "testing") for part in parts):
        return True
    basename = parts[-1]
    return (
        fnmatch.fnmatchcase(basename, "test_*.py")
        or fnmatch.fnmatchcase(basename, "*_test.py")
        or basename == "conftest.py"
    )


def diff_touches_test_path(diff_text):
    """True if ANY path in the diff is a test path (see is_test_path)."""
    return any(is_test_path(p) for p in _diff_touched_paths(diff_text))


# --- small IO helpers (deliberately NOT imported from export_trajectories —
# only the three names above are shared; these are local because the spec
# scopes `messages`'s user-turn source to tasks/dev/ specifically, unlike
# export_trajectories._read_task_description's dev-then-staging fallback) ---

def _read_lines(path):
    return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()


def _read_diff_text(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _read_task_description(tasks_root, task):
    """tasks/dev/<task>/task.yaml `description` — dev only (both runs this
    exporter draws from are 40-dev-task sweeps; holdout is sealed and never a
    training source, so it is never even looked at here)."""
    y = Path(tasks_root) / "dev" / task / "task.yaml"
    if not y.exists():
        return None
    spec = yaml.safe_load(y.read_text()) or {}
    return spec.get("description")


def _cached_task_description(tasks_root, task, cache):
    if task not in cache:
        cache[task] = _read_task_description(tasks_root, task)
    return cache[task]


def _run_variant(run_dir):
    """The variant name a run used, from its own summary.json (never assumed
    / hardcoded — the two default runs happen to use different variants:
    v001-baseline for the baseline run, v002-completion-contract for the
    recapture run)."""
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        sys.exit(f"run has no summary.json: {run_dir}")
    summary = json.loads(summary_path.read_text())
    variant = summary.get("variant")
    if not variant:
        sys.exit(f"run summary.json has no 'variant' field: {summary_path}")
    return variant


def _variant_system_text(variants_root, variant, cache):
    """experiments/variants/<variant>/claude-config/CLAUDE.md, cached by an
    explicit per-call dict (NOT a module-global / mutable-default cache) so
    repeated invocations — e.g. from separate unit tests reusing a variant
    NAME like "vtest" against different tmp_path fixture trees — never leak
    stale content across calls."""
    if variant not in cache:
        p = Path(variants_root) / variant / "claude-config" / "CLAUDE.md"
        if not p.exists():
            sys.exit(f"variant CLAUDE.md not found: {p}")
        cache[variant] = p.read_text()
    return cache[variant]


def _log(msg):
    print(msg, file=sys.stderr)


# --- length metrics --------------------------------------------------------

def _target_len(ex):
    """Cap-ordering length metric: mirrors export_trajectories.build_dataset
    exactly (serialized length of the model-produced turns only, i.e.
    messages[2:] — system/user are per-task-invariant so including them would
    not discriminate between a task's own trajectories)."""
    blob = json.dumps(ex["messages"][2:], sort_keys=True, ensure_ascii=False)
    return len(blob)


def _row_char_len(row):
    """Total content chars across ALL messages (system+user+assistant+tool) —
    the reporting metric for the length histogram, distinct from _target_len
    (which is cap-ordering only and excludes system/user by design)."""
    total = 0
    for m in row["messages"]:
        c = m["content"]
        total += len(c) if isinstance(c, str) else len(json.dumps(c, ensure_ascii=False))
    return total


def _percentile(sorted_values, p):
    """Linear-interpolation percentile (numpy-default-compatible) over an
    already-sorted sequence. p in [0, 1]. Empty input -> 0 (never raises)."""
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _length_stats(rows):
    lens = sorted(_row_char_len(r) / CHARS_PER_TOKEN for r in rows)
    return {
        "n": len(lens),
        "p50": round(_percentile(lens, 0.5), 1),
        "p90": round(_percentile(lens, 0.9), 1),
        "max": round(lens[-1], 1) if lens else 0,
    }


# --- core selection ---------------------------------------------------------

def build_preference_dataset(pos_run, neg_runs, tasks_root, variants_root,
                              out_path, pos_cap=3, neg_cap=3):
    """Build the preference JSONL from the given runs' trials.jsonl and write
    it to out_path. Returns the stats dict (also what the CLI prints).

    Selection rules (implemented exactly per the harness spec):
      - verdict == "invalid"                  -> excluded entirely, always.
      - verdict == "pass" AND run is pos_run  -> POSITIVE candidate, subject
        to: (a) the test-edit screen (worker.diff touches no test path), then
        (b) has_thinking (>=1 thinking char). Either failure excludes it.
      - verdict == "pass" AND run is NOT pos_run -> not eligible for anything
        (logged, not counted as one of the 5 core exclusion reasons).
      - verdict == "fail" AND run is a neg_run -> NEGATIVE candidate. NO
        test-edit screen (a gamed fail is still an undesirable fail). Missing
        worker.diff/transcript -> skip + log.
      - Per-task caps: positives keep the `pos_cap` SHORTEST (mirrors
        export_trajectories.build_dataset's shortest-first heuristic);
        negatives keep the `neg_cap` best by (reason == "empty_diff" first,
        then shortest) — empty_diff is the under-action target this whole
        rung exists to counteract.
    """
    pos_run = Path(pos_run)
    neg_runs = [Path(r) for r in neg_runs]
    tasks_root = Path(tasks_root)
    variants_root = Path(variants_root)

    pos_run_key = str(pos_run.resolve())
    neg_run_keys = {str(r.resolve()) for r in neg_runs}

    # De-dup runs to read (v002 is, by default, both the pos_run and one of
    # the neg_runs — its trials.jsonl must be read exactly once).
    runs_by_key = {}
    for r in [pos_run] + neg_runs:
        runs_by_key.setdefault(str(r.resolve()), r)

    system_text_cache = {}
    task_desc_cache = {}

    excluded = {
        "invalid": 0,
        "test_edit": 0,
        "no_thinking": 0,
        "missing_artifact": 0,
        "cap": 0,
        # Extra transparency buckets beyond the 5 core reasons: a trial whose
        # verdict structurally can never be used from the run it came from
        # (a pass outside the designated positive run; a fail from a run not
        # in --neg-run). Kept out of the 5 named reasons since these aren't
        # "candidate rejected on inspection" events — they're categorically
        # out of scope for that run by construction — but still logged so no
        # trial silently vanishes from the audit trail.
        "pass_ineligible_run": 0,
        "fail_ineligible_run": 0,
        "unknown_verdict": 0,
    }

    per_run_counts = {}
    pos_by_task = {}   # task -> [{"sort_key": length, "row": row}, ...]
    neg_by_task = {}   # task -> [{"sort_key": (prio, length), "reason":, "row":}, ...]

    for run_key, run_dir in runs_by_key.items():
        run_id = run_dir.name
        is_pos_run = run_key == pos_run_key
        is_neg_run = run_key in neg_run_keys

        variant = _run_variant(run_dir)
        system_text = _variant_system_text(variants_root, variant, system_text_cache)

        trials = list(_iter_objs(_read_lines(run_dir / "trials.jsonl")))
        vcounts = Counter()

        for t in trials:
            verdict = t.get("verdict")
            vcounts[verdict] += 1
            task = t.get("task")
            trial_num = t.get("trial")
            where = f"{run_id}/{task}/trial{trial_num}"

            if verdict == "invalid":
                excluded["invalid"] += 1
                _log(f"invalid (excluded entirely): {where}")
                continue

            if verdict == "pass":
                if not is_pos_run:
                    excluded["pass_ineligible_run"] += 1
                    _log(f"pass_ineligible_run: {where} (pass outside the "
                         "designated positive run — never a candidate)")
                    continue

                tpath = _local_transcript_path(run_dir, task, trial_num,
                                               t.get("transcript_path"))
                dpath = tpath.parent / "worker.diff"
                if not tpath.exists() or not dpath.exists():
                    excluded["missing_artifact"] += 1
                    _log(f"missing_artifact (pos): {where}")
                    continue

                if diff_touches_test_path(_read_diff_text(dpath)):
                    excluded["test_edit"] += 1
                    _log(f"test_edit (pos, excluded): {where}")
                    continue

                user_text = _cached_task_description(tasks_root, task, task_desc_cache)
                if user_text is None:
                    excluded["missing_artifact"] += 1
                    _log(f"missing_artifact (pos, no task.yaml): {where}")
                    continue

                ex = transcript_to_messages(_read_lines(tpath), system_text,
                                            user_text, task=task, trial=trial_num)
                if not ex["has_thinking"]:
                    excluded["no_thinking"] += 1
                    _log(f"no_thinking (pos, excluded): {where}")
                    continue

                row = {
                    "task": ex["task"], "trial": ex["trial"], "run_id": run_id,
                    "label": "pass", "messages": ex["messages"],
                    "n_assistant_turns": ex["n_assistant_turns"],
                    "n_tool_calls": ex["n_tool_calls"],
                    "has_thinking": ex["has_thinking"],
                    "thinking_chars": ex["thinking_chars"],
                }
                pos_by_task.setdefault(task, []).append(
                    {"sort_key": _target_len(ex), "row": row})

            elif verdict == "fail":
                if not is_neg_run:
                    excluded["fail_ineligible_run"] += 1
                    _log(f"fail_ineligible_run: {where} (fail outside any "
                         "--neg-run — never a candidate)")
                    continue

                tpath = _local_transcript_path(run_dir, task, trial_num,
                                               t.get("transcript_path"))
                dpath = tpath.parent / "worker.diff"
                if not tpath.exists() or not dpath.exists():
                    excluded["missing_artifact"] += 1
                    _log(f"missing_artifact (neg): {where}")
                    continue

                user_text = _cached_task_description(tasks_root, task, task_desc_cache)
                if user_text is None:
                    excluded["missing_artifact"] += 1
                    _log(f"missing_artifact (neg, no task.yaml): {where}")
                    continue

                # No test-edit screen for negatives (spec: a gamed fail is
                # still undesirable) and no has_thinking gate.
                ex = transcript_to_messages(_read_lines(tpath), system_text,
                                            user_text, task=task, trial=trial_num)
                row = {
                    "task": ex["task"], "trial": ex["trial"], "run_id": run_id,
                    "label": "fail", "messages": ex["messages"],
                    "n_assistant_turns": ex["n_assistant_turns"],
                    "n_tool_calls": ex["n_tool_calls"],
                    "has_thinking": ex["has_thinking"],
                    "thinking_chars": ex["thinking_chars"],
                }
                reason = t.get("reason")
                reason_priority = 0 if reason == "empty_diff" else 1
                neg_by_task.setdefault(task, []).append({
                    "sort_key": (reason_priority, _target_len(ex)),
                    "reason": reason, "row": row,
                })

            else:
                excluded["unknown_verdict"] += 1
                _log(f"unknown_verdict={verdict!r}: {where}")
                continue

        per_run_counts[run_id] = {
            "trials_total": len(trials),
            "verdict_counts": dict(vcounts),
            "role": {"positive_source": is_pos_run, "negative_source": is_neg_run},
        }

    # --- per-task caps ---
    kept_rows = []
    kept_neg_reasons = []  # parallel list of reason strings for kept negatives

    for task in sorted(pos_by_task):
        cands = sorted(pos_by_task[task], key=lambda c: c["sort_key"])
        keep, drop = cands[:pos_cap], cands[pos_cap:]
        excluded["cap"] += len(drop)
        for c in drop:
            r = c["row"]
            _log(f"cap (pos): dropped {r['run_id']}/{task}/trial{r['trial']} "
                 f"(target_len={c['sort_key']}, pos_cap={pos_cap})")
        kept_rows.extend(c["row"] for c in keep)

    for task in sorted(neg_by_task):
        cands = sorted(neg_by_task[task], key=lambda c: c["sort_key"])
        keep, drop = cands[:neg_cap], cands[neg_cap:]
        excluded["cap"] += len(drop)
        for c in drop:
            r = c["row"]
            _log(f"cap (neg): dropped {r['run_id']}/{task}/trial{r['trial']} "
                 f"(reason={c['reason']}, sort_key={c['sort_key']}, neg_cap={neg_cap})")
        for c in keep:
            kept_rows.append(c["row"])
            kept_neg_reasons.append(c["reason"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for run_id, rc in per_run_counts.items():
        rc["kept_pass"] = sum(1 for r in kept_rows
                              if r["run_id"] == run_id and r["label"] == "pass")
        rc["kept_fail"] = sum(1 for r in kept_rows
                              if r["run_id"] == run_id and r["label"] == "fail")

    pos_rows = [r for r in kept_rows if r["label"] == "pass"]
    neg_rows = [r for r in kept_rows if r["label"] == "fail"]

    stats = {
        "out": str(out_path),
        "pos_run": str(pos_run),
        "neg_runs": [str(r) for r in neg_runs],
        "pos_cap": pos_cap,
        "neg_cap": neg_cap,
        "totals_by_label": {
            "pass": len(pos_rows), "fail": len(neg_rows), "total": len(kept_rows),
        },
        "per_run_counts": per_run_counts,
        "excluded": excluded,
        "kept_negative_reason_mix": dict(Counter(kept_neg_reasons)),
        "length_histogram_chars_over_4": {
            "pass": _length_stats(pos_rows),
            "fail": _length_stats(neg_rows),
        },
    }
    return stats


# --- CLI ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--pos-cap", type=int, default=3,
                    help="max positives kept per task (default 3)")
    ap.add_argument("--neg-cap", type=int, default=3,
                    help="max negatives kept per task (default 3)")
    ap.add_argument("--pos-run", default=str(DEFAULT_POS_RUN),
                    help="run dir that may contribute POSITIVES "
                         f"(default: {DEFAULT_POS_RUN.name})")
    ap.add_argument("--neg-run", action="append", default=None,
                    help="run dir that may contribute NEGATIVES; repeatable "
                         "(default: both hardcoded runs — "
                         f"{DEFAULT_NEG_RUNS[0].name} and {DEFAULT_NEG_RUNS[1].name})")
    ap.add_argument("--tasks-root", default=str(ROOT / "tasks"))
    ap.add_argument("--variants-root", default=str(ROOT / "experiments" / "variants"))
    args = ap.parse_args()

    neg_runs = args.neg_run if args.neg_run else [str(p) for p in DEFAULT_NEG_RUNS]

    stats = build_preference_dataset(
        pos_run=args.pos_run,
        neg_runs=neg_runs,
        tasks_root=args.tasks_root,
        variants_root=args.variants_root,
        out_path=args.out,
        pos_cap=args.pos_cap,
        neg_cap=args.neg_cap,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
