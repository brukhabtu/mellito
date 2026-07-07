"""Unit tests for infra/sweep_stats.py — pure functions over trial-result
dicts, so no Modal/GPU/Docker needed. Formalizes the validation the module's
docstrings describe (see PLAN "Error != fail" for the invalid-trial rules).

Assertions here are deliberately subset-style (specific keys/values, not
whole-dict equality): another agent is concurrently adding an `api_usd` field
to cost() and a `worker_model` pass-through to summarize(), and this file must
keep passing once those land.
"""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sweep_stats import (
    _wilson,
    append_ledger,
    by_provenance,
    cost,
    paired,
    per_task,
    summarize,
    write_summary,
)


def _r(task, trial, verdict, provenance="public-pretrained", **kw):
    r = {
        "task": task, "trial": trial, "provenance": provenance,
        "verdict": verdict, "tokens_in": kw.pop("tokens_in", 10),
        "tokens_out": kw.pop("tokens_out", 20),
        "gpu_seconds": kw.pop("gpu_seconds", 5.0),
        "wall_clock_s": kw.pop("wall_clock_s", 30.0),
    }
    r.update(kw)
    return r


# ---------------------------------------------------------------- per_task --

def test_per_task_counts_and_pass_rate():
    results = [
        _r("A", 0, "pass"), _r("A", 1, "pass"), _r("A", 2, "fail"),
        _r("B", 0, "invalid"), _r("B", 1, "invalid"),
    ]
    pt = per_task(results)
    assert pt["A"]["passes"] == 2
    assert pt["A"]["valid"] == 3
    assert pt["A"]["invalid"] == 0
    assert pt["A"]["pass_rate"] == pytest.approx(2 / 3)

    assert pt["B"]["valid"] == 0
    assert pt["B"]["invalid"] == 2
    assert pt["B"]["pass_rate"] is None


def test_per_task_all_invalid_pass_rate_none():
    results = [_r("C", 0, "invalid"), _r("C", 1, "invalid"), _r("C", 2, "invalid")]
    pt = per_task(results)
    assert pt["C"]["pass_rate"] is None
    assert pt["C"]["valid"] == 0
    assert pt["C"]["invalid"] == 3


# ------------------------------------------------------------------ wilson --

def test_wilson_zero_trials():
    assert _wilson(0, 0) == (0.0, 0.0)


def test_wilson_k_equals_n_upper_bound_is_one():
    lo, hi = _wilson(5, 5)
    assert hi == 1.0
    assert 0.0 <= lo <= 1.0


def test_wilson_monotonic_in_k():
    # For fixed n, the interval (both bounds) should not decrease as k grows.
    n = 10
    prev_lo, prev_hi = _wilson(0, n)
    for k in range(1, n + 1):
        lo, hi = _wilson(k, n)
        assert lo >= prev_lo - 1e-9
        assert hi >= prev_hi - 1e-9
        prev_lo, prev_hi = lo, hi


def test_wilson_bounds_within_unit_interval():
    for k, n in [(0, 1), (1, 1), (3, 7), (100, 100)]:
        lo, hi = _wilson(k, n)
        assert 0.0 <= lo <= hi <= 1.0


# ------------------------------------------------------------------ paired --

def test_paired_no_parent():
    pt = per_task([_r("A", 0, "pass")])
    result = paired(pt, None)
    assert result["comparable_tasks"] == 0
    assert result["wins"] == 0
    assert result["losses"] == 0
    assert result["ties"] == 0
    assert result["net_tasks"] == 0
    assert "note" in result


def test_paired_win_loss_tie():
    # this variant: A yes, B no, C yes
    this_results = [
        _r("A", 0, "pass"), _r("A", 1, "pass"), _r("A", 2, "pass"),
        _r("B", 0, "fail"), _r("B", 1, "fail"), _r("B", 2, "fail"),
        _r("C", 0, "pass"), _r("C", 1, "pass"), _r("C", 2, "pass"),
    ]
    # parent variant: A yes, B yes, C no
    parent_results = [
        _r("A", 0, "pass"), _r("A", 1, "pass"), _r("A", 2, "pass"),
        _r("B", 0, "pass"), _r("B", 1, "pass"), _r("B", 2, "pass"),
        _r("C", 0, "fail"), _r("C", 1, "fail"), _r("C", 2, "fail"),
    ]
    this_pt = per_task(this_results)
    parent_pt = per_task(parent_results)
    result = paired(this_pt, parent_pt)
    assert result["wins"] == 1       # C: this solves, parent doesn't
    assert result["losses"] == 1     # B: parent solves, this doesn't
    assert result["ties"] == 1       # A: both solve
    assert result["net_tasks"] == 0
    assert result["comparable_tasks"] == 3


def test_paired_excludes_task_with_zero_valid_trials_on_one_side():
    this_results = [
        _r("A", 0, "pass"), _r("A", 1, "pass"), _r("A", 2, "pass"),
        # D: this has valid trials, parent has none (all invalid)
        _r("D", 0, "pass"),
    ]
    parent_results = [
        _r("A", 0, "pass"), _r("A", 1, "pass"), _r("A", 2, "pass"),
        _r("D", 0, "invalid"),
    ]
    this_pt = per_task(this_results)
    parent_pt = per_task(parent_results)
    result = paired(this_pt, parent_pt)
    # D excluded (parent has no valid trials for it); only A is comparable
    assert result["comparable_tasks"] == 1
    assert result["ties"] == 1
    assert result["wins"] == 0
    assert result["losses"] == 0


# ------------------------------------------------------------ by_provenance --

def test_by_provenance_groups_and_ci_bounds():
    results = [
        _r("A", 0, "pass", provenance="own-repo"),
        _r("A", 1, "pass", provenance="own-repo"),
        _r("A", 2, "pass", provenance="own-repo"),
        _r("B", 0, "fail", provenance="own-repo"),
        _r("B", 1, "fail", provenance="own-repo"),
        _r("B", 2, "fail", provenance="own-repo"),
        _r("C", 0, "pass", provenance="post-cutoff"),
        _r("C", 1, "pass", provenance="post-cutoff"),
        _r("C", 2, "pass", provenance="post-cutoff"),
    ]
    pt = per_task(results)
    bp = by_provenance(pt)

    assert set(bp.keys()) == {"own-repo", "post-cutoff"}
    assert bp["own-repo"]["tasks"] == 2
    assert bp["own-repo"]["valid_tasks"] == 2
    assert bp["own-repo"]["solved"] == 1  # A solved, B not
    assert bp["post-cutoff"]["solved"] == 1
    assert bp["post-cutoff"]["tasks"] == 1

    for stat in bp.values():
        lo, hi = stat["ci95"]
        assert 0.0 <= lo <= hi <= 1.0


def test_by_provenance_unknown_group_for_missing_provenance():
    results = [_r("A", 0, "pass", provenance=None)]
    pt = per_task(results)
    bp = by_provenance(pt)
    assert "unknown" in bp


# ------------------------------------------------------------------- cost --

def test_cost_basic_math_and_usd_rate():
    results = [
        _r("A", 0, "pass", gpu_seconds=3600.0, tokens_in=100, tokens_out=200,
           wall_clock_s=60.0),
        _r("A", 1, "pass", gpu_seconds=3600.0, tokens_in=50, tokens_out=50,
           wall_clock_s=30.0),
    ]
    c = cost(results, usd_per_gpu_hour=2.0)
    assert c["gpu_seconds"] == pytest.approx(7200.0)
    assert c["usd"] == pytest.approx(4.0)  # 2 gpu-hours * $2/hr
    assert c["tokens_in"] == 150
    assert c["tokens_out"] == 250
    assert c["trials"] == 2
    assert c["valid_trials"] == 2
    assert c["invalid_trials"] == 0
    assert c["mean_wall_s"] == pytest.approx(45.0)


def test_cost_invalid_trials_excluded_from_mean_wall_and_counted():
    results = [
        _r("A", 0, "pass", wall_clock_s=10.0),
        _r("A", 1, "invalid", wall_clock_s=9999.0),
        _r("A", 2, "invalid", wall_clock_s=9999.0),
    ]
    c = cost(results, usd_per_gpu_hour=1.0)
    assert c["invalid_trials"] == 2
    assert c["valid_trials"] == 1
    # mean_wall_s must be computed over valid trials only
    assert c["mean_wall_s"] == pytest.approx(10.0)


def test_cost_usd_per_solved_task_none_when_nothing_solved():
    results = [
        _r("A", 0, "fail"), _r("A", 1, "fail"), _r("A", 2, "fail"),
    ]
    c = cost(results, usd_per_gpu_hour=2.0)
    assert c["usd_per_solved_task"] is None


def test_cost_usd_per_solved_task_when_solved():
    results = [
        _r("A", 0, "pass", gpu_seconds=1800.0),
        _r("A", 1, "pass", gpu_seconds=1800.0),
    ]
    c = cost(results, usd_per_gpu_hour=2.0)
    assert c["usd_per_solved_task"] is not None
    assert c["usd_per_solved_task"] == pytest.approx(c["usd"] / 1)


# -------------------------------------------------------------- summarize --

def _three_task_results():
    # A: 3/3 pass -> solved
    # B: 1/3 pass -> not solved at 0.5 threshold
    # C: pass, invalid, pass -> 2 valid, 2 passes -> solved
    return [
        _r("A", 0, "pass"), _r("A", 1, "pass"), _r("A", 2, "pass"),
        _r("B", 0, "pass"), _r("B", 1, "fail"), _r("B", 2, "fail"),
        _r("C", 0, "pass"), _r("C", 1, "invalid"), _r("C", 2, "pass"),
    ]


def test_summarize_solved_and_valid_task_counts():
    results = _three_task_results()
    summary = summarize(
        run_id="run-001", variant="v001", parent=None,
        results=results, parent_per_task=None, usd_per_gpu_hour=1.0,
    )
    assert summary["run_id"] == "run-001"
    assert summary["variant"] == "v001"
    assert summary["tasks"] == 3
    assert summary["valid_tasks"] == 3
    assert summary["solved_tasks"] == 2  # A and C solved, B not
    assert summary["pass_rate_over_tasks"] == pytest.approx(2 / 3, abs=1e-3)
    assert "per_task" in summary
    assert set(summary["per_task"].keys()) == {"A", "B", "C"}
    assert summary["per_task"]["A"]["passes"] == 3
    assert summary["per_task"]["C"]["valid"] == 2
    assert summary["per_task"]["C"]["invalid"] == 1
    # nested blocks present and internally sane
    assert "paired_vs_parent" in summary
    assert "by_provenance" in summary
    assert "cost" in summary


# --------------------------------------------- write_summary / append_ledger --

def test_write_summary_round_trip(tmp_path):
    run_dir = tmp_path / "runs" / "run-xyz"
    summary = {"run_id": "run-xyz", "variant": "v002", "tasks": 1}
    p = write_summary(run_dir, summary)
    assert p == run_dir / "summary.json"
    assert p.exists()
    loaded = json.loads(p.read_text())
    assert loaded["run_id"] == "run-xyz"
    assert loaded["variant"] == "v002"


def test_append_ledger_header_written_once_and_row_format(tmp_path):
    ledger = tmp_path / "cost-ledger.csv"
    append_ledger(ledger, "run-1", "2026-07-07T00:00:00Z", 3600.0, 2.0)
    append_ledger(ledger, "run-2", "2026-07-07T01:00:00Z", 1800.0, 1.0)

    rows = list(csv.reader(ledger.open()))
    assert rows[0] == ["run_id", "timestamp", "gpu_seconds", "usd"]
    # header appears exactly once across the two appends
    assert sum(1 for row in rows if row == ["run_id", "timestamp", "gpu_seconds", "usd"]) == 1
    assert rows[1] == ["run-1", "2026-07-07T00:00:00Z", "3600.0", "2.0"]
    assert rows[2] == ["run-2", "2026-07-07T01:00:00Z", "1800.0", "1.0"]
    assert len(rows) == 3
