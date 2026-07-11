"""Unit tests for the P10.2 probe harness (pure logic, no Modal, no GPU)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import probe_harness as ph  # noqa: E402

TARGET = "csv-to-markdown"


# --- stream-json row builders --------------------------------------------

def _asst(content):
    return json.dumps({"type": "assistant", "message": {"role": "assistant",
                                                         "content": content}})


def _tool_use(name, inp):
    return {"type": "tool_use", "id": "tu_1", "name": name, "input": inp}


def _text(txt):
    return {"type": "text", "text": txt}


# --- detector -------------------------------------------------------------

def test_detect_s_channel_hit():
    lines = [
        json.dumps({"type": "system", "subtype": "init"}),
        _asst([_tool_use("SlashCommand", {"command": "/csv-to-markdown run"})]),
    ]
    d = ph.detect_invocation(lines, TARGET)
    assert d["S"] == 1 and d["K"] == 0
    assert d["invoked"] is True and d["channel"] == "S"


def test_detect_builtin_slashcommand_excluded():
    # Built-ins flow through the SAME SlashCommand tool; target-scoping must
    # not count them (pre-reg §2 / mechanism-notes §3).
    lines = [
        _asst([_tool_use("SlashCommand", {"command": "/compact"})]),
        _asst([_tool_use("SlashCommand", {"command": "/context"})]),
    ]
    d = ph.detect_invocation(lines, TARGET)
    assert d["S"] == 0 and d["invoked"] is False and d["channel"] == "none"


def test_detect_boundary_not_prefix_of_longer_name():
    # /csv-to-markdown-2 must NOT count as /csv-to-markdown.
    d = ph.detect_invocation(
        [_asst([_tool_use("SlashCommand", {"command": "/csv-to-markdown-2 x"})])],
        TARGET)
    assert d["S"] == 0 and d["invoked"] is False


def test_detect_k_channel_forward_compat():
    # Skill tool (forward-compat) naming the target -> counts, invoked=True.
    d1 = ph.detect_invocation(
        [_asst([_tool_use("Skill", {"command": "/csv-to-markdown"})])], TARGET)
    assert d1["K"] == 1 and d1["invoked"] is True and d1["channel"] == "K"
    # Bare name form {"name": "<target>"} also qualifies for channel K.
    d2 = ph.detect_invocation(
        [_asst([_tool_use("Skill", {"name": "csv-to-markdown"})])], TARGET)
    assert d2["K"] == 1 and d2["invoked"] is True


def test_detect_t_channel_not_counted_as_invocation():
    # Text mention only: T is reported but does NOT flip invoked.
    d = ph.detect_invocation(
        [_asst([_text("I could use /csv-to-markdown for this.")])], TARGET)
    assert d["T"] == 1 and d["S"] == 0 and d["K"] == 0
    assert d["invoked"] is False and d["channel"] == "T"


def test_detect_no_invocation():
    d = ph.detect_invocation(
        [_asst([_text("Here is your table.")]),
         _asst([_tool_use("Bash", {"command": "echo hi"})])], TARGET)
    assert d == {"S": 0, "K": 0, "T": 0, "invoked": False, "channel": "none"}


def test_detect_malformed_rows_tolerated():
    lines = [
        "",                                   # blank
        "not json at all {",                  # garbage
        json.dumps([1, 2, 3]),                # non-dict JSON
        json.dumps({"type": "assistant"}),    # no message/content
        json.dumps({"type": "assistant", "message": {"content": "oops"}}),  # str content
        _asst(["not-a-block", {"type": "tool_use"}]),  # bad blocks
        _asst([_tool_use("SlashCommand", {"command": "/csv-to-markdown go"})]),  # real hit
    ]
    d = ph.detect_invocation(lines, TARGET)  # must not raise
    assert d["S"] == 1 and d["invoked"] is True


def test_detect_accepts_string_transcript():
    text = _asst([_tool_use("SlashCommand", {"command": "/csv-to-markdown"})])
    assert ph.detect_invocation(text, TARGET)["invoked"] is True


# --- workspace generation -------------------------------------------------

def test_eleven_grid_cells_present():
    grid = ph.grid_cells()
    assert len(grid) == 11
    assert ph.HOT_CELL_ID in grid and ph.COLD_CELL_ID in grid
    # forced cell is extra, not part of the scoring grid.
    assert "forced-invocation" not in grid
    assert "forced-invocation" in ph.all_cells()


def test_all_eleven_cells_materialize(tmp_path):
    for cid, cell in ph.grid_cells().items():
        dest = tmp_path / cid
        prov = ph.materialize_probe(cell, 0, dest)
        assert (dest / "prompt.txt").read_text() == cell.prompt
        assert (dest / "cell.json").exists()
        skill_dirs = sorted(p.name for p in (dest / ".claude" / "skills").iterdir())
        assert len(skill_dirs) == cell.n_distractors + 1
        # Every skill dir has a SKILL.md with valid-looking frontmatter.
        for sd in skill_dirs:
            md = (dest / ".claude" / "skills" / sd / "SKILL.md").read_text()
            assert md.startswith("---\n") and "\nname: " in md and "description:" in md
        # provenance matches the materialized tree.
        assert prov["cell_id"] == cid
        assert len(prov["skills"]) == cell.n_distractors + 1
        assert prov["seed"] == f"{cid}:0"


def test_single_skill_cell_uses_bare_dirname(tmp_path):
    cell = ph.get_cell(ph.HOT_CELL_ID)
    ph.materialize_probe(cell, 0, tmp_path)
    dirs = [p.name for p in (tmp_path / ".claude" / "skills").iterdir()]
    assert dirs == ["csv-to-markdown"]  # no numeric prefix when solo


def test_c20_distractor_descriptions_exceed_500_chars(tmp_path):
    cell = ph.get_cell("C-20-last")
    assert cell.n_distractors == 19
    prov = ph.materialize_probe(cell, 0, tmp_path)
    distractors = [s for s in prov["skills"] if s["role"] == "distractor"]
    assert len(distractors) == 19
    for s in distractors:
        assert s["desc_len"] >= ph.C20_MIN_DESC_CHARS
        assert s["desc_len"] <= ph.LISTING_MAX_DESC_CHARS
    # Target sits last and keeps its normal (B-specific, ~30-word) description.
    target = [s for s in prov["skills"] if s["role"] == "target"][0]
    assert target["position"] == 19 and target["desc_len"] < ph.C20_MIN_DESC_CHARS


def test_position_randomization_deterministic_under_seed():
    cell = ph.get_cell("C-20-last")
    a = [it["spec"].name for it in ph.ordered_skills(cell, 3)]
    b = [it["spec"].name for it in ph.ordered_skills(cell, 3)]
    assert a == b                       # same trial -> identical order
    c = [it["spec"].name for it in ph.ordered_skills(cell, 4)]
    assert a != c                       # different trial -> shuffled differently
    # Target position is FIXED (last) regardless of trial.
    assert a[-1] == TARGET and c[-1] == TARGET


def test_target_position_first_vs_last():
    first = ph.ordered_skills(ph.get_cell("C-5-first"), 0)
    last = ph.ordered_skills(ph.get_cell("C-5-last"), 0)
    assert first[0]["spec"].name == TARGET and first[0]["role"] == "target"
    assert last[-1]["spec"].name == TARGET and last[-1]["role"] == "target"
    assert len(first) == 5 and len(last) == 5


def test_multi_skill_dirs_carry_ordering_prefix(tmp_path):
    ph.materialize_probe(ph.get_cell("C-5-first"), 0, tmp_path)
    dirs = sorted(p.name for p in (tmp_path / ".claude" / "skills").iterdir())
    # Directory names are position-prefixed so filesystem order == listing order.
    assert dirs[0] == "00-csv-to-markdown"
    assert all(d[:2].isdigit() and d[2] == "-" for d in dirs)


def test_d_necessity_cell_shape(tmp_path):
    cell = ph.get_cell("D-necessity")
    assert cell.target_skill == "ticket-id-format"
    assert cell.records_solved is True and cell.solvable_without is False
    prov = ph.materialize_probe(cell, 0, tmp_path)
    body = (tmp_path / ".claude" / "skills" / "ticket-id-format" / "SKILL.md").read_text()
    assert "epoch-week" in body   # load-bearing, unrecoverable-without-body info
    assert prov["records_solved"] is True


def test_cell_json_provenance_roundtrips(tmp_path):
    cell = ph.get_cell("C-5-last")
    ph.materialize_probe(cell, 2, tmp_path)
    prov = json.loads((tmp_path / "cell.json").read_text())
    assert prov["cell_id"] == "C-5-last"
    assert prov["trial_idx"] == 2
    assert prov["factors"] == {"A": "high", "B": "specific",
                               "C": "5-last", "D": "convenience"}
    assert prov["target_position"] == "last"
    assert [s["position"] for s in prov["skills"]] == [0, 1, 2, 3, 4]


# --- gate evaluation ------------------------------------------------------

def test_gate_pass():
    g = ph.manipulation_gate(hot_k=9, hot_n=10, cold_k=1, cold_n=10)
    assert g["status"] == "pass"
    assert g["hot"]["rate"] == 0.9 and g["cold"]["rate"] == 0.1


def test_gate_non_degenerate_miss():
    # Clear separation, hot > cold, but 70/30 misses both thresholds.
    g = ph.manipulation_gate(hot_k=7, hot_n=10, cold_k=3, cold_n=10)
    assert g["status"] == "non_degenerate_miss"


def test_gate_degenerate_no_separation():
    g = ph.manipulation_gate(hot_k=3, hot_n=10, cold_k=5, cold_n=10)
    assert g["status"] == "degenerate" and "<= cold" in g["reason"]


def test_gate_degenerate_both_floor():
    g = ph.manipulation_gate(hot_k=2, hot_n=10, cold_k=1, cold_n=10)
    assert g["status"] == "degenerate" and "floor" in g["reason"]


def test_gate_degenerate_both_ceiling():
    g = ph.manipulation_gate(hot_k=10, hot_n=10, cold_k=9, cold_n=10)
    assert g["status"] == "degenerate" and "ceiling" in g["reason"]


# --- CI sanity (known values) --------------------------------------------

def test_wilson_known_values():
    lo, hi = ph.wilson_ci(8, 10)
    assert round(lo, 3) == 0.490 and round(hi, 3) == 0.943
    assert ph.wilson_ci(0, 0) == (0.0, 0.0)
    assert ph.wilson_ci(0, 10)[0] == 0.0
    assert ph.wilson_ci(10, 10)[1] == 1.0


def test_newcombe_known_value():
    # Hand-computed (module docstring formula) for 8/10 vs 2/10:
    #   Wilson(8,10)=(0.49016,0.94333), Wilson(2,10)=(0.05667,0.50984), diff=0.6
    #   lower=0.6-sqrt(0.30984^2+0.30984^2)=0.1618
    #   upper=0.6+sqrt(0.14333^2+0.14333^2)=0.8027
    lo, hi = ph.newcombe_diff_ci(8, 10, 2, 10)
    assert abs(lo - 0.1618) < 0.002
    assert abs(hi - 0.8027) < 0.002


def test_newcombe_symmetric_for_identical_samples():
    lo, hi = ph.newcombe_diff_ci(5, 10, 5, 10)
    assert abs((hi + lo)) < 1e-9      # symmetric around diff=0
    assert lo < 0 < hi


# --- runner guard ---------------------------------------------------------

def test_run_probe_local_refuses_without_auth(tmp_path, monkeypatch):
    monkeypatch.delenv("PROBE_AUTH_OK", raising=False)
    try:
        ph.run_probe_local(ph.get_cell(ph.HOT_CELL_ID), 0, "sonnet", tmp_path)
        assert False, "should have refused without PROBE_AUTH_OK"
    except RuntimeError as e:
        assert "PROBE_AUTH_OK" in str(e)


def test_run_probe_ornith_is_plan_only(tmp_path):
    try:
        ph.run_probe_ornith(ph.get_cell(ph.HOT_CELL_ID), 0, tmp_path)
        assert False, "should be plan-only"
    except NotImplementedError:
        pass
