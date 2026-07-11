"""P10.2 skill-invocation probe harness — pure logic, no Modal imports.

Builds the pre-registered probe grid (findings/p10-probe-grid-draft.md, locked
by OPUS 2026-07-11), materializes isolated probe workspaces as DATA under
`experiments/probes/<run_id>/<cell>/<trial>/`, detects model-driven skill
invocation from a Claude Code `stream-json` transcript, and aggregates
per-cell rates + the two-cell manipulation-check gate.

Structural invariant (CLAUDE.md): a probe workspace's `.claude/skills/` is the
SUBJECT — generated data, exactly like `experiments/variants/*/claude-config/`.
It is never this repo's own `.claude/`, and nothing here writes into it.

What lives here (all pure / stdlib-only, unit-testable without a GPU):
  - Cell registry: the 11 locked grid cells + a proof-of-one `forced` cell.
  - `materialize_probe(cell, trial_idx, dest_dir)` — writes a probe workspace.
  - `detect_invocation(transcript_lines, target_skill)` — the dual-channel,
    target-scoped detector (channels S / K / T) over the same stream-json
    shape `trial_logic.parse_stream_json` consumes.
  - Aggregation: Wilson per-cell CIs, Newcombe difference CI, and the literal
    80/20 manipulation-check gate with the pass / non-degenerate-miss /
    degenerate classification.

Runner adapters (`run_probe_local`, `run_probe_ornith`) live here too but are
NOT executed by any test: `run_probe_local` is guarded behind PROBE_AUTH_OK=1
(the operator auth gate), and the Ornith adapter is a documented P10.4 plan
that raises NotImplementedError — modal_app.py is untouched until P10.4.

DETECTOR NOTE (deviation from mechanism-notes §3, per the P10.2 build brief):
the binary primary outcome `invoked` is `S ∨ K` only. Channel T (a bare text
mention of `/<target>`) is detected and reported but does NOT flip `invoked`.
mechanism-notes §3 / pre-reg §2 fold T into `invoked`; the build brief scopes T
to "tertiary, reported but not counted as invocation" and pins a test to it.
The build brief governs the implementation; the discrepancy is surfaced, not
silently resolved.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# Probe workspaces are generated here; only experiments/probes/README.md is
# committed (see that file). This module never writes outside the dest_dir it
# is handed, so the run_id/cell/trial layout is imposed by the caller.
PROBE_ROOT = "experiments/probes"

# Short per pre-reg §2 ("single-turn-or-few-turn"): probes carry no repo
# context, so a handful of turns is ample; a large cap only invites drift.
PROBE_MAX_TURNS = 6

# Manipulation-check gate thresholds (pre-reg §4, LOCKED, literal 80/20). Also
# reused as the near-floor / near-ceiling anchors for the degeneracy check.
GATE_HOT_MIN = 0.80
GATE_COLD_MAX = 0.20

# The C-20 truncation cell mandates distractor descriptions >= this many chars
# so the listing overflows the 1%-of-context budget (pre-reg §1 Factor C, §7.4).
C20_MIN_DESC_CHARS = 500
# Claude-Code combined description+when_to_use listing cap (mechanism-notes §2).
LISTING_MAX_DESC_CHARS = 1536


# --------------------------------------------------------------------------
# Skill + cell data model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillSpec:
    """One SKILL.md to materialize: frontmatter `name` + `description`, body."""
    name: str
    description: str
    body: str
    role: str = "target"  # "target" | "distractor"


@dataclass(frozen=True)
class Cell:
    """One pre-registered probe cell. `factors` records the A/B/C/D levels; the
    remaining fields are the materialization recipe derived from them."""
    cell_id: str
    factors: dict
    target_skill: str
    prompt: str
    n_distractors: int
    target_position: str          # "solo" | "first" | "last"
    distractor_desc_mode: str     # "none" | "short" | "long"
    solvable_without: bool        # is the task solvable without invoking?
    records_solved: bool          # D-necessity cells log a secondary `solved`
    is_forced: bool = False       # auth-free proof-of-one cell (not in grid)
    is_stretch: bool = False      # B-first-person stretch cell
    notes: str = ""


# --------------------------------------------------------------------------
# Literal strings from the pre-registration (§1)
# --------------------------------------------------------------------------

_CSV_DESC = {
    "specific": (
        "Convert a CSV file to formatted Markdown table syntax. Use when the "
        "user asks to convert a CSV to Markdown, turn CSV data into a Markdown "
        "table, or format tabular data as Markdown."
    ),
    "vague": "Helps with data formatting.",
    "first-person": "I can help you convert CSV files into Markdown tables.",
}

_CSV_BODY = (
    "To convert CSV to a Markdown table: treat the first row as the header, "
    "emit a separator row of dashes (one per column), then one table row per "
    "remaining CSV line. Separate cells with the pipe character `|`."
)

_TICKET_DESC = (
    "Apply this repo's internal ticket-ID format. Use when the user asks to "
    "generate, format, or validate a ticket ID for this project."
)

_TICKET_BODY = (
    "Ticket IDs use the format `<team-code>-<epoch-week>-<3-digit-sequence>`, "
    "e.g. `INFRA-2938-014`. team-code is one of ENG/INFRA/DATA/OPS. epoch-week "
    "is the ISO week number since 2020-01-01 (not calendar week). The 3-digit "
    "sequence resets every epoch-week and must be zero-padded."
)

# Task prompts by Factor-A level (§1 Factor A). The CSV literal keeps the \n
# escapes shown in the pre-reg verbatim inside backticks.
_PROMPTS = {
    "high": r"Convert this CSV to a Markdown table: `name,age\nAda,30\nGrace,29`",
    "medium": (
        "I have some comma-separated data — name,age with two rows — can you "
        "lay it out as a table I can paste into a `.md` file?"
    ),
    "low": (
        "Write a one-paragraph summary of why binary search requires a sorted "
        "input array."
    ),
}
_TICKET_PROMPT = "Generate a valid ticket ID for the INFRA team for today."
_FORCED_PROMPT = (
    "Invoke /csv-to-markdown now, then use it to convert this CSV to a "
    r"Markdown table: `name,age\nAda,30\nGrace,29`"
)


# Distractor pool (§1 Factor C): unrelated-domain, B-specific-quality skills.
# The first 4 seed the C-5 cells; the first 19 seed C-20-last. Each entry is
# (name, short third-person description with explicit "Use when..." triggers).
_DISTRACTOR_POOL = [
    ("git-commit-helper",
     "Write clear conventional-commit messages from a staged diff. Use when "
     "the user asks to write a commit message, summarize staged changes, or "
     "format a commit in Conventional Commits style."),
    ("json-validator",
     "Validate JSON text against a schema and report the first error. Use when "
     "the user asks to check if JSON is valid, validate a JSON payload, or find "
     "a syntax error in JSON."),
    ("unit-converter",
     "Convert between physical units of length, mass, temperature, and volume. "
     "Use when the user asks to convert units, change miles to kilometers, or "
     "express a measurement in metric."),
    ("regex-tester",
     "Test a regular expression against sample strings and explain each match "
     "group. Use when the user asks to test a regex, check what a pattern "
     "matches, or debug a regular expression."),
    ("yaml-linter",
     "Lint YAML for indentation and duplicate-key errors. Use when the user "
     "asks to validate YAML, check a YAML config, or find why a YAML file fails "
     "to parse."),
    ("sql-formatter",
     "Reformat a SQL query with consistent keyword casing and indentation. Use "
     "when the user asks to format SQL, pretty-print a query, or clean up a "
     "SELECT statement."),
    ("color-picker",
     "Convert colors between HEX, RGB, and HSL and suggest accessible "
     "contrasts. Use when the user asks to convert a color, get an RGB value, "
     "or check color contrast."),
    ("timezone-converter",
     "Convert a timestamp between time zones with DST handling. Use when the "
     "user asks to convert a time between zones, find a UTC offset, or schedule "
     "across regions."),
    ("base64-codec",
     "Encode or decode Base64 text and detect malformed padding. Use when the "
     "user asks to encode Base64, decode a Base64 string, or fix Base64 "
     "padding."),
    ("uuid-generator",
     "Generate and validate UUIDs of versions 1, 4, and 7. Use when the user "
     "asks to generate a UUID, create a unique id, or validate a UUID string."),
    ("markdown-linter",
     "Check Markdown for heading order and broken link syntax. Use when the "
     "user asks to lint Markdown, validate a README, or find Markdown "
     "formatting issues."),
    ("cron-explainer",
     "Explain a cron expression in plain English and show the next run times. "
     "Use when the user asks what a cron expression means, decode a crontab "
     "line, or schedule a cron job."),
    ("http-status-lookup",
     "Explain HTTP status codes and their typical causes. Use when the user "
     "asks what an HTTP status means, why a request returns 429, or how to "
     "handle a 503."),
    ("semver-bumper",
     "Compute the next semantic version from a change type. Use when the user "
     "asks to bump a version, decide major/minor/patch, or validate a semver "
     "string."),
    ("gitignore-builder",
     "Assemble a .gitignore from language and tool templates. Use when the user "
     "asks to create a gitignore, ignore build artifacts, or add IDE files to "
     "gitignore."),
    ("dockerfile-linter",
     "Lint a Dockerfile for cache-busting and security anti-patterns. Use when "
     "the user asks to review a Dockerfile, optimize image layers, or fix a "
     "Docker build."),
    ("env-var-auditor",
     "Find undocumented or unused environment variables in a project. Use when "
     "the user asks to audit env vars, list required environment variables, or "
     "check a .env file."),
    ("changelog-writer",
     "Draft a Keep-a-Changelog entry from merged pull requests. Use when the "
     "user asks to write a changelog, summarize a release, or format release "
     "notes."),
    ("license-picker",
     "Recommend an open-source license from project constraints. Use when the "
     "user asks which license to choose, compare MIT and Apache, or add a "
     "LICENSE file."),
    ("dependency-grapher",
     "Render a dependency graph from a lockfile. Use when the user asks to "
     "visualize dependencies, find a transitive dependency, or detect a "
     "dependency cycle."),
]


def _long_description(name: str, short: str) -> str:
    """Deterministic >=500-char (<= listing cap) description for the C-20 cell.

    Pre-reg §1 Factor C mandates distractor descriptions >= ~500 chars (pushed
    toward the 1,536-char cap) so twenty entries overflow the ~1%-of-context
    listing budget and truncation engages. This expands a short description into
    a well-formed, still third-person, trigger-rich one — the "generation rule"
    the build brief allows in place of 19 hand-written long strings.
    """
    topic = name.replace("-", " ")
    extra = (
        f" This skill handles {topic} end to end, covering the everyday cases a "
        f"developer meets as well as the awkward edge cases that are easy to get "
        f"wrong by hand. It states explicit trigger phrases so the model can tell "
        f"when it applies: use it when the user mentions {topic}, asks to run a "
        f"{topic} operation, pastes input that looks like a {topic} problem, or "
        f"requests any step the {topic} workflow is known to cover. It documents "
        f"the inputs it expects and the output it produces, states its "
        f"assumptions plainly, and calls out the failure modes it guards against "
        f"so the result is predictable and straightforward to verify against the "
        f"user's original request without further back-and-forth."
    )
    return (short + extra)[:LISTING_MAX_DESC_CHARS]


# --------------------------------------------------------------------------
# Cell registry (§3 / §4)
# --------------------------------------------------------------------------

HOT_CELL_ID = "A-high_B-specific"
COLD_CELL_ID = "A-low_B-specific"


def _grid_cells() -> "dict[str, Cell]":
    """The 11 LOCKED cells (pre-reg §3 table): the A{3}×B{2} core block, the C
    sweep (3 new; C-solo == the A-high×B-specific core cell), the D-necessity
    cell, and the B-first-person stretch cell."""
    cells: list[Cell] = []

    # --- Core A×B factorial (C=solo, D=convenience) -----------------------
    for a in ("high", "medium", "low"):
        for b in ("specific", "vague"):
            cells.append(Cell(
                cell_id=f"A-{a}_B-{b}",
                factors={"A": a, "B": b, "C": "solo", "D": "convenience"},
                target_skill="csv-to-markdown",
                prompt=_PROMPTS[a],
                n_distractors=0,
                target_position="solo",
                distractor_desc_mode="none",
                solvable_without=True,
                records_solved=False,
                notes="core A×B factorial cell",
            ))

    # --- C sweep (A=high, B=specific, D=convenience) ----------------------
    # C-solo is the A-high×B-specific core cell above — not duplicated here.
    cells.append(Cell(
        cell_id="C-5-first",
        factors={"A": "high", "B": "specific", "C": "5-first", "D": "convenience"},
        target_skill="csv-to-markdown", prompt=_PROMPTS["high"],
        n_distractors=4, target_position="first", distractor_desc_mode="short",
        solvable_without=True, records_solved=False,
        notes="target lists first among 5 well-written skills",
    ))
    cells.append(Cell(
        cell_id="C-5-last",
        factors={"A": "high", "B": "specific", "C": "5-last", "D": "convenience"},
        target_skill="csv-to-markdown", prompt=_PROMPTS["high"],
        n_distractors=4, target_position="last", distractor_desc_mode="short",
        solvable_without=True, records_solved=False,
        notes="target lists last among 5 well-written skills",
    ))
    cells.append(Cell(
        cell_id="C-20-last",
        factors={"A": "high", "B": "specific", "C": "20-last", "D": "convenience"},
        target_skill="csv-to-markdown", prompt=_PROMPTS["high"],
        n_distractors=19, target_position="last", distractor_desc_mode="long",
        solvable_without=True, records_solved=False,
        notes=("count/dilution + truncation-PRESENCE only; zero invocation "
               "history makes the documented drop-order uncontrollable, so this "
               "is NOT a clean target-drop manipulation (pre-reg §1 Factor C). "
               "Validity precondition: /context must confirm the listing "
               "overflowed the budget and >=1 description was dropped."),
    ))

    # --- D sweep (A=high, B=specific, C=solo): only D-necessity is new -----
    cells.append(Cell(
        cell_id="D-necessity",
        factors={"A": "high", "B": "specific", "C": "solo", "D": "necessity"},
        target_skill="ticket-id-format", prompt=_TICKET_PROMPT,
        n_distractors=0, target_position="solo", distractor_desc_mode="none",
        solvable_without=False, records_solved=True,
        notes="load-bearing skill body; secondary `solved` outcome recorded",
    ))

    # --- B-first-person stretch (A=high, C=solo, D=convenience) ------------
    cells.append(Cell(
        cell_id="B-first-person",
        factors={"A": "high", "B": "first-person", "C": "solo", "D": "convenience"},
        target_skill="csv-to-markdown", prompt=_PROMPTS["high"],
        n_distractors=0, target_position="solo", distractor_desc_mode="none",
        solvable_without=True, records_solved=False, is_stretch=True,
        notes="third-person contract violation; tests if the rule is load-bearing",
    ))

    return {c.cell_id: c for c in cells}


# The auth-free proof-of-one cell (build brief item 2c): an explicit
# "invoke /<target> now" instruction that pins the SlashCommand `input` shape
# cheaply with EITHER subject. NOT part of the 11-cell grid or the gate.
_FORCED_CELL = Cell(
    cell_id="forced-invocation",
    factors={"A": "high", "B": "specific", "C": "solo", "D": "convenience"},
    target_skill="csv-to-markdown", prompt=_FORCED_PROMPT,
    n_distractors=0, target_position="solo", distractor_desc_mode="none",
    solvable_without=True, records_solved=False, is_forced=True,
    notes=("proof-of-one: explicit invoke instruction; use to pin the "
           "SlashCommand input shape auth-free before scoring any grid cell"),
)

_GRID = _grid_cells()
_ALL = dict(_GRID)
_ALL[_FORCED_CELL.cell_id] = _FORCED_CELL


def grid_cells() -> "dict[str, Cell]":
    """The 11 locked scoring cells, keyed by cell_id (excludes the forced cell)."""
    return dict(_GRID)


def all_cells() -> "dict[str, Cell]":
    """The 11 grid cells plus the auth-free `forced-invocation` proof-of-one cell."""
    return dict(_ALL)


def get_cell(cell_id: str) -> Cell:
    return _ALL[cell_id]


# --------------------------------------------------------------------------
# Skill-spec construction
# --------------------------------------------------------------------------


def _target_spec(cell: Cell) -> SkillSpec:
    if cell.target_skill == "ticket-id-format":
        return SkillSpec("ticket-id-format", _TICKET_DESC, _TICKET_BODY, "target")
    # csv-to-markdown: description varies with Factor B, body is constant.
    desc = _CSV_DESC[cell.factors["B"]]
    return SkillSpec("csv-to-markdown", desc, _CSV_BODY, "target")


def _distractor_specs(cell: Cell) -> "list[SkillSpec]":
    specs = []
    for name, short in _DISTRACTOR_POOL[: cell.n_distractors]:
        desc = _long_description(name, short) if cell.distractor_desc_mode == "long" else short
        body = f"Apply the standard {name.replace('-', ' ')} procedure described above."
        specs.append(SkillSpec(name, desc, body, "distractor"))
    return specs


def ordered_skills(cell: Cell, trial_idx: int) -> "list[dict]":
    """The skills of `cell` in listing order for one trial, positions assigned.

    Distractor internal order is shuffled per trial with a seed derived from
    (cell_id, trial_idx) — string-seeded RNG is stable across interpreter runs
    (independent of PYTHONHASHSEED), so a run is reproducible. The TARGET's
    position (first/last) is the manipulated variable and is FIXED within a cell
    across its N trials (pre-reg §2 "Position randomization").

    Returns [{"position": int, "spec": SkillSpec, "role": str}, ...].
    """
    target = _target_spec(cell)
    distractors = _distractor_specs(cell)
    if not distractors:
        return [{"position": 0, "spec": target, "role": target.role}]

    rng = random.Random(f"{cell.cell_id}:{trial_idx}")
    rng.shuffle(distractors)
    if cell.target_position == "first":
        order = [target] + distractors
    else:  # "last"
        order = distractors + [target]
    return [{"position": i, "spec": s, "role": s.role} for i, s in enumerate(order)]


def _skill_md(spec: SkillSpec) -> str:
    """SKILL.md text: YAML frontmatter (name + description) then a body.

    `name` is set explicitly so the invocable form stays `/<name>` regardless of
    the directory's ordering prefix. `description` is emitted as a JSON string,
    which is valid YAML for any scalar (handles backticks, commas, quotes)."""
    return (
        "---\n"
        f"name: {spec.name}\n"
        f"description: {json.dumps(spec.description)}\n"
        "---\n\n"
        f"# {spec.name}\n\n"
        f"{spec.body}\n"
    )


def _skill_dirname(position: int, name: str, multi: bool) -> str:
    """Directory name for a skill. Multi-skill cells prefix a 2-digit position
    so filesystem ordering follows the designed listing order; single-skill
    cells use the bare name. (Whether the harness lists by directory name or by
    frontmatter `name` is exactly what the proof-of-one confirms — the ordering
    is also recorded in cell.json so analysis never depends on inferring it.)"""
    return f"{position:02d}-{name}" if multi else name


# --------------------------------------------------------------------------
# Workspace generator
# --------------------------------------------------------------------------


def materialize_probe(cell: Cell, trial_idx: int, dest_dir) -> dict:
    """Write an isolated probe workspace for one trial under `dest_dir`.

    Layout (DATA — under experiments/probes/<run_id>/<cell>/<trial>/ at run time):
      <dest>/.claude/skills/<dir>/SKILL.md   one per installed skill
      <dest>/prompt.txt                      the task prompt
      <dest>/cell.json                       provenance record

    Returns the provenance dict (also written to cell.json).
    """
    dest = Path(dest_dir)
    skills_root = dest / ".claude" / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)

    order = ordered_skills(cell, trial_idx)
    multi = len(order) > 1
    skills_prov = []
    for item in order:
        spec = item["spec"]
        dirname = _skill_dirname(item["position"], spec.name, multi)
        sdir = skills_root / dirname
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "SKILL.md").write_text(_skill_md(spec))
        skills_prov.append({
            "position": item["position"],
            "dir": dirname,
            "name": spec.name,
            "role": spec.role,
            "desc_len": len(spec.description),
        })

    (dest / "prompt.txt").write_text(cell.prompt)

    provenance = {
        "cell_id": cell.cell_id,
        "trial_idx": trial_idx,
        "factors": cell.factors,
        "target_skill": cell.target_skill,
        "target_position": cell.target_position,
        "n_distractors": cell.n_distractors,
        "distractor_desc_mode": cell.distractor_desc_mode,
        "solvable_without": cell.solvable_without,
        "records_solved": cell.records_solved,
        "is_forced": cell.is_forced,
        "is_stretch": cell.is_stretch,
        "seed": f"{cell.cell_id}:{trial_idx}",
        "prompt": cell.prompt,
        "skills": skills_prov,
        "notes": cell.notes,
    }
    (dest / "cell.json").write_text(json.dumps(provenance, indent=2))
    return provenance


# --------------------------------------------------------------------------
# Invocation detector (mechanism-notes §3, pre-reg §2/§7.1)
# --------------------------------------------------------------------------


def _target_re(target: str) -> "re.Pattern":
    """Match `/<target>` at a token boundary — excludes built-ins (/compact,
    /context, …) by requiring the exact skill name, and excludes longer names
    (`/csv-to-markdown-2`) by forbidding a trailing word char or hyphen."""
    return re.compile(r"/" + re.escape(target) + r"(?![\w-])")


def _command_strings(inp) -> "list[str]":
    """Candidate command strings from a tool_use `input`. The exact key holding
    the `/<name>` string on the pinned 2.0.14 SlashCommand tool is finalized by
    the proof-of-one; until then we prefer the documented `command` key and fall
    back to other plausible keys / any string value, so the detector plumbing is
    robust to the observed shape without over-counting (the target-boundary
    match below is what actually gates a hit)."""
    if isinstance(inp, str):
        return [inp]
    if not isinstance(inp, dict):
        return []
    out = []
    for key in ("command", "slash_command", "name", "skill", "prompt", "text", "args"):
        v = inp.get(key)
        if isinstance(v, str):
            out.append(v)
    # Any remaining string values (defensive against an unexpected key name).
    for v in inp.values():
        if isinstance(v, str) and v not in out:
            out.append(v)
    return out


def _names_target(inp, rx: "re.Pattern", target: str) -> bool:
    """True iff a tool_use input names the target skill — either as a
    `/<target>` command token or as a bare skill name equal to the target."""
    for s in _command_strings(inp):
        st = s.strip()
        if rx.match(st) or rx.search(st):
            return True
        if st == target:  # e.g. Skill tool with input {"name": "csv-to-markdown"}
            return True
    return False


def _assistant_content(obj: dict) -> list:
    """Content blocks of an `assistant` stream-json row, tolerating shapes."""
    msg = obj.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
    else:
        content = obj.get("content")
    if isinstance(content, list):
        return content
    return []


def detect_invocation(transcript_lines, target_skill: str) -> dict:
    """Dual-channel, target-scoped invocation detector over a stream-json
    transcript (list of lines or a single string).

    Channels (mechanism-notes §3):
      S  tool_use name == "SlashCommand" whose input command targets /<target>
         (primary on the pinned Claude Code 2.0.14; built-ins excluded by the
         target-boundary scoping).
      K  tool_use name == "Skill" whose input names the target (forward-compat;
         expected 0 on 2.0.14).
      T  assistant TEXT block containing the literal /<target> token (tertiary).

    Returns {"S", "K", "T", "invoked", "channel"} where `invoked = S>0 or K>0`
    (channel T is reported but does NOT flip `invoked` — see module docstring).
    Malformed / partial / non-dict rows are tolerated and skipped.
    """
    if isinstance(transcript_lines, str):
        lines = transcript_lines.splitlines()
    else:
        lines = transcript_lines or []

    rx = _target_re(target_skill)
    s = k = t = 0
    for line in lines:
        line = (line or "").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict) or obj.get("type") != "assistant":
            continue
        for block in _assistant_content(obj):
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                name = block.get("name")
                inp = block.get("input")
                if name == "SlashCommand" and _names_target(inp, rx, target_skill):
                    s += 1
                elif name == "Skill" and _names_target(inp, rx, target_skill):
                    k += 1
            elif btype == "text":
                txt = block.get("text")
                if isinstance(txt, str) and rx.search(txt):
                    t += 1

    invoked = (s > 0) or (k > 0)
    channel = "S" if s > 0 else ("K" if k > 0 else ("T" if t > 0 else "none"))
    return {"S": s, "K": k, "T": t, "invoked": invoked, "channel": channel}


# --------------------------------------------------------------------------
# Aggregation: Wilson / Newcombe CIs + the manipulation gate (pre-reg §6/§4)
# --------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> "tuple[float, float]":
    """95% Wilson score interval for k successes of n trials (matches
    sweep_stats._wilson; stays sane at small N and near 0/1)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe_diff_ci(k1: int, n1: int, k2: int, n2: int,
                     z: float = 1.96) -> "tuple[float, float]":
    """Newcombe hybrid-score 95% CI for the difference p1 - p2 (method 10,
    "square-and-add"): the pre-reg's chosen small-sample CI on a difference of
    proportions (§6.2), no scipy needed. With l_i/u_i the Wilson bounds:
        lower = (p1-p2) - sqrt((p1-l1)^2 + (u2-p2)^2)
        upper = (p1-p2) + sqrt((u1-p1)^2 + (p2-l2)^2)
    """
    p1 = (k1 / n1) if n1 else 0.0
    p2 = (k2 / n2) if n2 else 0.0
    l1, u1 = wilson_ci(k1, n1, z)
    l2, u2 = wilson_ci(k2, n2, z)
    diff = p1 - p2
    lower = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lower), min(1.0, upper))


def cell_rate(k: int, n: int) -> dict:
    """Per-cell invocation-rate summary: `k/N (rate%, [lo%, hi%])` (pre-reg §6.1)."""
    lo, hi = wilson_ci(k, n)
    return {
        "k": k, "n": n,
        "rate": (k / n) if n else None,
        "ci95": [round(lo, 4), round(hi, 4)],
    }


def manipulation_gate(hot_k: int, hot_n: int, cold_k: int, cold_n: int) -> dict:
    """Evaluate the LOCKED literal 80/20 manipulation-check gate (pre-reg §4).

    Status (the binding pass/fail line; the Newcombe gap-CI is reported only to
    classify a miss, never to override the literal line):
      pass                 hot rate >= 0.80 AND cold rate <= 0.20  → run the grid
      non_degenerate_miss  clear separation (hot > cold) but a threshold missed
                           → one redesign allowed, then re-gate (PLAN branch)
      degenerate           hot <= cold, OR both near-floor, OR both near-ceiling
                           → the kill condition after one sanctioned redesign
    """
    hot = cell_rate(hot_k, hot_n)
    cold = cell_rate(cold_k, cold_n)
    hr, cr = hot["rate"], cold["rate"]
    gap_lo, gap_hi = newcombe_diff_ci(hot_k, hot_n, cold_k, cold_n)

    if hr is not None and cr is not None and hr >= GATE_HOT_MIN and cr <= GATE_COLD_MAX:
        status, reason = "pass", "hot>=80% and cold<=20%"
    elif hr is None or cr is None or hr <= cr:
        status, reason = "degenerate", "hot rate <= cold rate (no separation)"
    elif hr <= GATE_COLD_MAX and cr <= GATE_COLD_MAX:
        status, reason = "degenerate", "both cells near floor"
    elif hr >= GATE_HOT_MIN and cr >= GATE_HOT_MIN:
        status, reason = "degenerate", "both cells near ceiling"
    else:
        status, reason = "non_degenerate_miss", "separation present, threshold missed"

    return {
        "status": status,
        "reason": reason,
        "hot": hot,
        "cold": cold,
        "gap": (hr - cr) if (hr is not None and cr is not None) else None,
        "gap_newcombe_ci95": [round(gap_lo, 4), round(gap_hi, 4)],
        "thresholds": {"hot_min": GATE_HOT_MIN, "cold_max": GATE_COLD_MAX},
    }


# --------------------------------------------------------------------------
# Runner adapters — BUILT, not executed by any test
# --------------------------------------------------------------------------


def run_probe_local(cell: Cell, trial: int, model: str, workdir,
                    *, max_turns: int = PROBE_MAX_TURNS, timeout_s: int = 300) -> dict:
    """Run ONE probe trial with a local headless `claude -p` in a materialized
    workspace, then run the detector. Reference-subject adapter for P10.2/P10.3.

    GUARDED: refuses unless env PROBE_AUTH_OK=1 — the operator auth gate. Auth
    for reference subjects is an open operator decision (PLAN §P10.3); this
    adapter must not fire a billed session until the operator sets that flag.

    The manipulation check is launched by calling this over the hot then cold
    cell at N=10 each; see experiments/probes/README.md and the module for the
    exact two-command invocation.
    """
    if os.environ.get("PROBE_AUTH_OK") != "1":
        raise RuntimeError(
            "run_probe_local refused: reference-subject auth is an open operator "
            "decision (PLAN §P10.3). Set PROBE_AUTH_OK=1 to authorize.")

    dest = Path(workdir)
    provenance = materialize_probe(cell, trial, dest)

    cmd = [
        "claude", "-p", cell.prompt,
        "--output-format", "stream-json", "--verbose",
        "--model", model,
        "--dangerously-skip-permissions",
        "--max-turns", str(max_turns),
    ]
    # IS_SANDBOX=1 lets --dangerously-skip-permissions run non-interactively
    # (same reason trial_logic.worker_env sets it); stdin from /dev/null avoids
    # the open-non-TTY-stdin hang.
    proc = subprocess.run(
        cmd, cwd=str(dest), capture_output=True, text=True,
        timeout=timeout_s, stdin=subprocess.DEVNULL,
        env={**os.environ, "IS_SANDBOX": "1"},
    )
    transcript = proc.stdout
    (dest / "transcript.jsonl").write_text(transcript)
    detection = detect_invocation(transcript.splitlines(), cell.target_skill)

    return {
        "cell_id": cell.cell_id,
        "trial": trial,
        "model": model,
        "returncode": proc.returncode,
        "detection": detection,
        "provenance": provenance,
        "transcript_path": str(dest / "transcript.jsonl"),
    }


def run_probe_ornith(cell: Cell, trial: int, workdir, **kwargs) -> dict:
    """P10.4 Ornith adapter — DOCUMENTED PLAN ONLY, not implemented here.

    Plan (lands in P10.4; modal_app.py is NOT modified before then):

    Probes ride the existing Modal sandbox machinery in `run_trial`. Where
    run_trial today materializes a variant's `claude-config/` into the task
    workspace's `.claude/` (modal_app.py ~L847-851: write the tar, untar, then
    `cp -a claude-config/. /testbed/.claude/`), the probe path instead
    materializes THIS module's probe workspace: build the workspace with
    `materialize_probe(cell, trial, dest)` locally, tar its `.claude/` skills
    tree, ship it to the sandbox, and untar into `/testbed/.claude/`. The task
    prompt (`cell.prompt`) is written to `/testbed/TASK.md` and passed as the
    worker prompt — exactly the WORKER_PROMPT channel run_trial already uses.

    The worker `claude` exec runs under `trial_logic.worker_env(...)` with
    ANTHROPIC_BASE_URL pointed at the in-app LiteLLM proxy and
    ANTHROPIC_MODEL=<ornith> — the same serving stack the baseline used — and
    emits a stream-json transcript. That transcript is pulled back and fed to
    `detect_invocation(lines, cell.target_skill)`; there is NO verifier/diff
    phase (probes have no code change), only the detector + (for D-necessity)
    the secondary `solved` check.

    Cost is warm-H100 wall-clock bound, not per-token (pre-reg §5b); N and
    concurrency are P10.4's own pre-registration, re-derived from a proof-of-one
    on the pinned Ornith container. No GPU work runs here.
    """
    raise NotImplementedError(
        "run_probe_ornith is a P10.4 plan (see docstring); modal_app.py is not "
        "modified until P10.4.")
