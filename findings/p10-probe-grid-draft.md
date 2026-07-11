# P10.2 — Probe grid pre-registration (DRAFT — for OPUS review, not yet committed)

Status: **DRAFT**. Written by SONNET per P10.1 (PLAN.md §P10 phases). To be
reviewed by a second (OPUS) agent, then committed by the orchestrator before
P10.2 build starts. Nothing in this file has been run. No probe workspace,
task, or skill exists yet on disk — `experiments/probes/` does not exist and
is not created by this document.

Mechanism grounding: `findings/p10-mechanism-notes.md` (P10.1 companion
deliverable) — read it first; the detector spec, the frontmatter contract,
and the open questions listed there are load-bearing for the design below.

Grid dimensions and cost figures below are proposals for OPUS to accept,
adjust, or reject before P10.2 build begins.

## 0. Structural invariants (restated, binding)

- Probe workspaces are **generated** under `experiments/probes/<probe_id>/`
  at P10.2 build time. They are DATA, exactly like
  `experiments/variants/*/claude-config/` — never this repo's own
  `.claude/`, never copied into it.
- `tasks/holdout/` is not touched by this work; if any probe generator
  accidentally references it, that is a guard-hook block, logged, not
  worked around (per experiment-integrity rule).
- No Modal GPU calls in P10.1/P10.2 reference-model work. Ornith cost math
  below is a projection for P10.4, included because it was asked for, not
  a commitment to run it here.

## 1. Factors and levels

Four factors, per PLAN.md §North-star goal P10-A. Each cell below is a
concrete SKILL.md `description` + probe task prompt pair, not a
placeholder — these are the literal strings P10.2 should generate (or
near-verbatim variants) for the two extreme manipulation-check cells (§4);
the middle levels are sketched for the full grid (§3).

### Factor A — task↔description semantic overlap

Holds description specificity, skill count(=1), and necessity(=convenience)
fixed; varies only how closely the task text's vocabulary/intent matches
the description's stated trigger conditions.

- **A-high (verbatim/near-verbatim overlap).**
  `description`: *"Convert a CSV file to formatted Markdown table syntax.
  Use when the user asks to convert a CSV to Markdown, turn CSV data into a
  Markdown table, or format tabular data as Markdown."*
  Task prompt: *"Convert this CSV to a Markdown table: `name,age\nAda,30\nGrace,29`"*
  — shares "CSV", "Markdown", "table," "convert" directly with the
  description.
- **A-medium (paraphrase/synonym overlap).**
  Same description. Task prompt: *"I have some comma-separated data —
  name,age with two rows — can you lay it out as a table I can paste into
  a `.md` file?"* — same intent, no shared trigger tokens ("comma-separated"
  not "CSV", "lay it out as a table" not "convert to Markdown table",
  "`.md` file" not "Markdown").
- **A-low (unrelated).**
  Same description. Task prompt: *"Write a one-paragraph summary of why
  binary search requires a sorted input array."* — no semantic relationship
  to CSV/Markdown/table conversion at all; the skill should not fire and a
  correct model should solve the task without it.

### Factor B — description specificity / trigger phrasing

Holds task (fixed at the A-high CSV-to-Markdown prompt above), skill
count(=1), and necessity(=convenience) fixed; varies only the
`description` field's adherence to the documented authoring guidance
(mechanism-notes §2).

- **B-specific.** The A-high description above: third person, states what
  the skill does AND when to use it, includes concrete trigger words the
  task can match ("CSV", "Markdown table", "convert").
- **B-vague.** `description`: *"Helps with data formatting."* — the exact
  shape of the anti-pattern the best-practices doc calls out by name
  ("Helps with documents" / "Processes data").
- **B-first-person (contract violation, secondary probe).**
  `description`: *"I can help you convert CSV files into Markdown
  tables."* — tests whether the documented "always third person" rule is
  load-bearing for selection or merely a style preference; cheap to add
  since it reuses the B-specific task/skill body otherwise unchanged.

### Factor C — skill count + list position

Holds task (A-high CSV prompt), the target skill's description
(B-specific), and necessity(=convenience) fixed; varies how many *other*
skills are installed alongside the target and where the target sits in the
resulting listing. Distractor skills are unrelated-domain, equally
well-written (B-specific quality) skills, so any effect is attributable to
count/position rather than distractor quality.

- **C-solo.** Only the target skill installed (count = 1; no position
  variable).
- **C-5-first.** Target skill + 4 unrelated well-written distractors
  (e.g. `git-commit-helper`, `json-validator`, `unit-converter`,
  `regex-tester`); target's directory is ordered/named so it lists first.
- **C-5-last.** Same 5 skills; target ordered last.
- **C-20-last.** Target + 19 unrelated well-written distractors; target
  last. Intended to probe the truncation regime (mechanism-notes §4) — but
  **truncation is not automatically reachable at 20 skills, and the default
  ~30-word descriptions used elsewhere in this grid do NOT reach it.**
  Explicit arithmetic (documented budget = 1% of the subject's context
  window; per-entry cap 1,536 chars ≈ ~384 tokens; ~4 chars/token):
  - Subject context window **200k tokens** (standard Claude Sonnet/Opus) →
    budget ≈ **2,000 tokens ≈ 8,000 chars**. Twenty ~30-word
    (~200-char ≈ ~50-token) entries = ~1,000 tokens < 2,000 → **no
    truncation** (the whole listing fits with margin).
  - To force overflow at 20 skills on a 200k subject, mean entry must
    exceed 2,000/20 = **100 tokens ≈ ~400 chars**. Distractor descriptions
    are therefore **mandated ≥ ~500 chars each**, ideally pushed toward the
    1,536-char per-entry cap; at the cap, 20 × 384 = ~7,680 tokens ≫ 2,000,
    truncation strongly engaged.
  - **If the subject runs at a 1M context window**, budget ≈ 10,000 tokens
    ≈ 40,000 chars; even 20 near-cap (1,536-char) entries = ~30,720 chars
    still fit → truncation UNREACHABLE at 20 skills, requiring ~27+ skills.
    So the subject's context window is a **build-time input** that sets the
    required count × description length.
  - **Build-time precondition (binding):** the cell is valid only if
    `/context` (documented to report post-budget listing size) confirms the
    listing actually exceeds the budget and at least one description was
    dropped. If it cannot be made to engage within the count/length above,
    **relabel this cell "count/dilution only, truncation not exercised"**
    and drop the truncation claim (see §3 note).
  - **Interpretation caveat (documented drop-order):** truncation drops
    "the skills the *user* invokes least" — i.e. by **historical
    invocation frequency across sessions** (mechanism-notes §4). In a
    fresh single-session probe workspace **every skill has zero invocation
    history**, so *which* description is dropped is **not controllable** and
    the docs do not specify the zero-history tiebreak. Therefore C-20-last
    can measure (i) count/dilution and (ii) whether truncation ENGAGES AT
    ALL (observable via `/context`), but it **cannot cleanly attribute a
    drop to the target** or test "target's description was truncated →
    invocation fell." Report it as a mechanism-presence check, not a clean
    target-drop manipulation.

`C-20-first` is optional/stretch (see §5 budget) — the core design covers
count(1,5) fully crossed with position, plus count=20 at one position, on
the logic that count=1 has no position and count=20-first mainly serves as
a truncation-mechanism check rather than a distinct scientific cell.

### Factor D — necessity vs. convenience of skill content

Holds task-family, description (B-specific), and skill count(=1) fixed;
varies whether the SKILL.md body carries information the task genuinely
cannot be completed without ("need-engineered", the same admission bar
P10.5's corpus will use) vs. information that is merely a shortcut to a
solution reachable without it.

- **D-convenience.** The A-high CSV-to-Markdown skill/task above — any
  competent model can hand-format a 2-row CSV as a Markdown table without
  reading the skill body; the skill only saves a little thought.
- **D-necessity.** `description`: *"Apply this repo's internal ticket-ID
  format. Use when the user asks to generate, format, or validate a ticket
  ID for this project."* SKILL.md body: *"Ticket IDs use the format
  `<team-code>-<epoch-week>-<3-digit-sequence>`, e.g. `INFRA-2938-014`.
  team-code is one of ENG/INFRA/DATA/OPS. epoch-week is the ISO week
  number since 2020-01-01 (not calendar week). The 3-digit sequence resets
  every epoch-week and must be zero-padded."* Task prompt: *"Generate a
  valid ticket ID for the INFRA team for today."* — the epoch-week
  definition and zero-padding rule are arbitrary and unrecoverable without
  reading the body; a model that never invokes the skill cannot produce a
  correct answer by any amount of reasoning, satisfying the same
  admission bar corpus-curator uses for need-engineered tasks (skill
  carries load-bearing info absent from task text; VERIFY requires it).

## 2. Design

- **Probe unit**: one short, single-turn-or-few-turn headless worker
  session per trial, in an isolated workspace under
  `experiments/probes/<probe_id>/trial_<n>/`, with a skills directory
  installed per the cell's spec (§1) and a `TASK.md` holding the task
  prompt. No repo context beyond the minimal skill(s) + task — these are
  *not* SWE-bench-style repo tasks; they are short enough that a model
  could plausibly solve most of them in one or two turns, keeping token
  cost and detector ambiguity low.
- **Outcome variable**: binary `invoked ∈ {0, 1}` per trial, per the
  dual-channel detector in mechanism-notes §3. The pinned harness is Claude
  Code `2.0.14`, which emits the **`SlashCommand`** tool (verified: no
  `Skill` tool exists in the pin's uniform init tool list) — so the primary
  signal (channel S) is a `tool_use` with `name == "SlashCommand"` whose
  `input.command` begins with `/<target-skill-name>` (the argument scope is
  mandatory — bare `SlashCommand` calls include built-ins like `/compact`
  and must not be counted). Channel K (`name == "Skill"`, forward-compat,
  expected 0 on `2.0.14`) and channel T (assistant text containing the
  literal `/<target-skill-name>`) are also detected; **`invoked` = S ∨ K
  only** — an actual tool call. Channel T is REPORTED per cell but never
  flips the binary: mentioning a skill in prose is not invoking it.
  *(AMENDED 2026-07-11, before any probe data existed: the original text
  read "`invoked` = any channel names the target ≥ once", which would have
  counted bare text mentions as invocations. Reconciled to the
  implementation at P10.2 build review; see FINDINGS P10.2 entry.)*
  **Per-channel counts (S/K/T) are reported per cell.** P10.2's first build step is the mandatory proof-of-one trial that
  pins the exact `SlashCommand` `input` shape before any grid cell runs for
  score (mechanism-notes §3, "Mandatory proof-of-one").
- **N per cell**: proposed **N=10** for the manipulation check (§4) and
  **N=15** for the full P10.3 grid (§5) — see cost math for how these were
  set against budget, not the other way around. These are small; §6 is
  explicit about what that does and does not license.
- **Session prompt held constant**: identical system-prompt scaffolding
  (a minimal "resolve this task" instruction, no mention of skills or any
  hint to use them) across every cell and every subject model. Only the
  installed skill set and the task text vary. This isolates the
  progressive-disclosure mechanism from prompt-level nudging — if the
  session prompt told the model to check its skills, that would confound
  every cell identically and the grid would measure compliance with an
  instruction, not spontaneous disclosure-driven invocation.
- **Temperature/determinism**: `temperature` is not sent at all on
  current-generation models per the reference-model docs (adaptive
  thinking; sampling params are rejected on Opus 4.7+/Sonnet 5, and
  omitting them is the supported path) — so trial-to-trial variation
  comes from the model's own decoding stochasticity, not a tunable knob.
  N independent trials per cell is the only lever for estimating the
  invocation rate's sampling variance; there is no "run once at
  temperature=0" shortcut available on the reference-model tier this grid
  targets.
- **Position randomization**: within Factor C's multi-skill cells, the
  *distractor* skills' internal order is randomized per trial (fresh
  shuffle each trial) so a distractor-ordering artifact cannot masquerade
  as a target-position effect; only the target skill's position
  (first/last) is the manipulated variable and is held fixed within a
  cell across its N trials.
- **Task solvability without the skill (control)**: recorded as a
  property of every task at design time, not just Factor D's two levels —
  A/B/C cells are all D-convenience (solvable without invocation) by
  construction, so a non-invoking trial in those cells is not
  automatically a "failure," only a non-invocation; only D-necessity
  cells make solvability the discriminating outcome, and D-necessity
  trials additionally record a secondary `solved ∈ {0,1}` (VERIFY-style
  check against the arbitrary format) to support the P10-C-style
  invocation-vs-cash-out distinction even at this pilot scale.
- **Description length as a controlled (not free) variable**: B-specific
  and B-vague descriptions in §1 are deliberately similar in length
  (~30 words each) so that the specific/vague contrast isolates content
  quality, not verbosity. If OPUS wants an explicit length sub-probe, the
  cheapest addition is a B-specific-short vs. B-specific-long pair at
  fixed content quality — flagged here as a stretch cell, not in the core
  budget.

## 3. Full grid shape (P10.3)

Rather than a full 3×3×4×2 = 72-cell factorial (infeasible at ≤$10), the
design is a **reduced grid**: one core A×B factorial at fixed
C=solo/D=convenience, plus two single-factor sweeps that vary C and D
independently off a fixed A-high/B-specific baseline.

**LOCKED (OPUS decision): the 11-cell reduced grid stands.** It satisfies
the P10-A gate literally — the gate asks for "≥4 pre-registered factors …
each with effect estimates," i.e. **marginal** per-factor effects, and the
design delivers a risk-difference + Newcombe CI for each of A, B, C, D
(§6). Interactions are NOT required by the gate. Two points make the lock
the honest call rather than a budget dodge:
- **The one interaction with both plausible main effects is already free.**
  The core block is a full **A{3}×B{2} factorial**, so the
  **overlap×specificity** interaction — the most decision-relevant one,
  since both factors bear on "does the model see a matching description" —
  is estimable (descriptively) from the existing 6 cells at **zero added
  cost**. It is not a missing capability; §6 reports the 2×3 cell means so
  it is inspectable.
- **C- and D-involving interactions are deliberately deferred, not bought.**
  The C and D sweeps are marginal-only (fixed A-high/B-specific baseline),
  so e.g. overlap×position or necessity×overlap are not estimable here.
  Adding them (a 2-cell A-low×position pair, or a necessity×A-low pair)
  is affordable on the budget (§5) but would yield a **difference-in-
  differences of proportions at N=15**, whose Newcombe CI spans well beyond
  the ±30-point practical-significance bar (§6) — i.e. it would be
  uninterpretable by this design's own analysis plan, buying the
  *appearance* of an interaction estimate without the power to use it, and
  spending budget on numbers §6 could not read. If an interaction proves
  worth the power, it belongs in a **sequential follow-up pre-registration
  after the main effects are in hand**, sized to detect it — not bolted on
  here. This keeps every cell interpretable as one factor's marginal
  effect, matching the "keep it simple and honest" instruction.

| Block | Cells | Factors varied | Factors held fixed |
|---|---|---|---|
| Core A×B | A{high,medium,low} × B{specific,vague} = 6 | overlap, specificity | C=solo, D=convenience |
| C sweep | C{solo, 5-first, 5-last, 20-last} = 4 | count/position | A=high, B=specific, D=convenience |
| D sweep | D{convenience, necessity} = 2 (convenience cell reused from Core A×B's A-high×B-specific cell — not re-run) | necessity | A=high, B=specific, C=solo |
| B-first-person stretch | 1 | contract violation | A=high, C=solo, D=convenience |

Distinct cells requiring their own trials: 6 (core) + 3 (C sweep, since
C-solo is shared with core's A-high×B-specific cell) + 1 (D-necessity,
new) + 1 (B-first-person, stretch) = **11 cells** at N=15 = 165 trials
(core), or 12 with the stretch cell = 180 trials.

## 4. Manipulation check — the two extreme cells (P10.2 gate)

Pre-registered **before** any full-grid trial runs, per PLAN.md P10-A gate
("the probe harness shows non-degenerate variance on a 2-cell manipulation
check BEFORE the full grid").

- **Hot cell** (obviously-should-invoke): A-high overlap × B-specific ×
  C-solo × D-convenience — the CSV-to-Markdown example in §1, verbatim.
  This is the single most favorable cell for invocation the grid contains
  (exact keyword match, textbook-quality description, no competing
  skills, no truncation pressure).
- **Cold cell** (obviously-should-not-invoke): A-low overlap (binary-search
  explanation task) × the same CSV-to-Markdown skill installed
  (B-specific, C-solo, D-convenience) — the skill is present and
  well-described, but nothing about the task should trigger it. This is
  a stronger test than "no skills installed at all," because it confirms
  the model is discriminating on relevance rather than simply never
  invoking anything.

**Pre-registered pass condition (LOCKED — OPUS decision): the literal
`≥80% hot AND ≤20% cold`, at N=10/cell, is the BINDING gate.** The
Newcombe gap-CI is reported alongside as a secondary diagnostic but is
**not** the gate. Rationale (the decision, made on the merits):

- **Legibility and non-gameability.** The gate is a pre-registered
  pass/fail line the operator can check in one glance; a literal
  `≥8/10 hot AND ≤2/10 cold` cannot be re-argued after the fact. A CI on
  the hot−cold gap invites exactly the post-hoc latitude the brief warns
  about ("does this 40-pt gap with a CI barely excluding 0 count?") and, at
  **N=10, the Newcombe CI on a difference of proportions is ~±30-40 points
  wide** — too fragile to bear the weight of a go/no-go on the *whole*
  characterization phase. The point estimates need not be precise; the gate
  only needs the two extreme cells to be unambiguously separated, which
  8/10-vs-2/10 delivers.
- **It operationalizes the PLAN kill criterion cleanly.** PLAN's kill
  trigger is "manipulation check floor/ceiling in **both** extreme cells →
  one redesign, then kill" — i.e. degeneracy. Mapping to the literal gate:
  - **PASS** (≥80% hot AND ≤20% cold) → proceed to the full grid.
  - **Non-degenerate miss** (clear separation, hot > cold, but a threshold
    is missed — e.g. 70%/30%) → **one redesign allowed** per PLAN, then
    re-gate; this is the "one redesign" branch, not an immediate kill.
  - **Degenerate** (both cells near-floor, both near-ceiling, or
    hot ≤ cold) → the kill condition; after the one sanctioned redesign,
    kill characterization and write up.
  The gap-CI is reported only to *classify* which branch a miss falls into
  (separation present vs. degenerate), never to override the literal line.
- **N can be raised before locking if the proof-of-one shows cheap tokens.**
  At N=10 the gate needs 8/10 and 2/10; the §5 fallback of N=20/cell
  ($0.80, still under the ≤$1 P10.2 cap) tightens both Wilson intervals and
  is the recommended bump if the proof-of-one confirms per-trial cost is at
  or below the §5 estimate. The 80/20 thresholds do not change with N; only
  the required counts (16/20 and 4/20) do.

**Cell-reuse note:** the hot and cold cells ARE two of the six core A×B
cells (A-high×B-specific and A-low×B-specific, both at C-solo/D-convenience).
The P10.2 manipulation check runs them at N=10 under their own run IDs as
the gate. Because the detector/config may be adjusted by P10.2's
proof-of-one, P10.3 **re-runs these two cells fresh at the full N=15**
rather than pooling across a possible config change (cost impact is
negligible — 2 cells × 5 extra trials). Pooling is permitted ONLY if the
CLI version, config, and detector are byte-identical between the two phases,
which must be asserted in FINDINGS before any pooling.

## 5. Cost math

### (a) Reference model via API

Current per-million-token pricing (fetched via the `claude-api` skill,
cache dated 2026-06-24): Claude Haiku 4.5 $1.00/$5.00 (in/out), Claude
Sonnet 5 $3.00/$15.00 ($2.00/$10.00 intro through 2026-08-31), Claude Opus
4.8 $5.00/$25.00, all per MTok. **Sonnet 5 is proposed as the P10.3
reference subject** — it is the mid-tier model the docs explicitly warn
behaves differently from Haiku on skill triggering ("what works for Opus
might need more detail for Haiku"), and its cost is affordable at this
scale; Haiku is proposed only as an optional cheap secondary pass if
budget remains, not as the primary subject, since a floor-tier model
could plausibly show near-zero invocation for capability reasons
unrelated to the mechanism being tested (confounding the grid).

Per-trial estimate: these are short, 1-2 turn probes (§2) — no repo
context, no file exploration. Assume ~2,000–4,000 input tokens (skill
listing + task text + one round of tool_result if the skill is invoked)
and ~500–1,000 output tokens (short reasoning + either the direct answer
or a `Skill` tool_use plus a short follow-up). At the high end of that
range (4,000 in / 1,000 out) on Sonnet 5 at intro pricing: 4,000/1e6 ×
$2.00 + 1,000/1e6 × $10.00 = $0.008 + $0.010 = **$0.018/trial**, rounded up
to **$0.02/trial** for a safety margin (retries, occasional longer
reasoning, prompt-caching misses since each probe workspace's skill
listing differs cell-to-cell and caching benefit across trials within a
cell is not assumed).

- **P10.2 manipulation check**: 2 cells × N=10 = 20 trials × $0.02 =
  **$0.40**, comfortably under the $1 cap and leaving room to raise N to
  ~20/cell ($0.80 total) if OPUS wants a tighter CI before greenlighting
  the full grid — proposed as the fallback if the N=10 CIs in §4 are
  judged too wide to trust.
- **P10.3 full grid**: 11 distinct cells × N=15 = 165 trials × $0.02 =
  **$3.30**. This leaves roughly $6.70 of the $10 cap unused at the
  proposed N — deliberately conservative, both because probe token counts
  are an estimate before any real trial has run, and to leave headroom
  for the manipulation check's own spend (§4) and for re-running any
  cell that comes back with an ambiguous or clearly-broken result (e.g.
  all-invalid trials from a malformed probe workspace) without blowing
  the cap. If P10.2's proof-of-one trial shows token counts well under
  this estimate, N per cell can be raised before P10.3 locks its design
  (e.g. N=25-30 would still fit under $10 at the same per-trial cost).

### (b) Ornith via the existing `serve()` stack

Different cost structure — Ornith is billed by warm H100-seconds on
Modal, not per-token API pricing, and the dominant cost driver observed
in this repo to date has been keeping a container warm across a sweep
rather than per-trial token volume (FINDINGS 2026-07-08: attributed rate
observed at **~$3.95/H100-hr**, run `20260707T215242-v001-baseline` at
$0.086/solved on 40×5 dev tasks; the throughput-tuning entries in
FINDINGS/PLAN Status report generation speed lifted from ~13 tok/s to
~908 tok/s aggregate under concurrency). Probe tasks are far shorter than
the 40-task SWE-bench dev set that baseline measured, so per-trial token
cost is a rounding error against the container's per-hour rate: at
~900 tok/s aggregate throughput, even a generous 5,000-token trial
(prompt processing + generation) costs on the order of a few seconds of
attributed H100 time, i.e. a few thousandths of a dollar. **The binding
constraint for P10.4 is therefore warm-container wall-clock, not the
$/trial math** — at $3.95/hr, the ≤$5 GPU cap for P10.4 (PLAN §P10 phases)
buys roughly **76 minutes of warm H100 time**, which at the throughput
figures already demonstrated in this repo is enough to run the full
11-cell × N grid (165+ trials) with room for cold-boot overhead, provided
trials are batched with reasonable concurrency rather than run one at a
time serially. This P10.1 note does not commit to a specific N for P10.4
— that is P10.4's own pre-registration, owned by whichever session runs
it, and should re-derive N from the actual proof-of-one timing on the
pinned Ornith container rather than this estimate.

## 6. Analysis plan

Given the small N per cell (10-15) and the project's own standing
discipline about not over-trusting small-sample deltas (PLAN.md §Decision
rules — the dev-corpus MDD language is the closest precedent even though
it's phrased for paired task counts, not proportions), the analysis
commits to being descriptive first, inferential only where the effect is
large enough that a crude interval already excludes zero:

1. **Per-cell invocation rate** with a **Wilson score 95% CI** (chosen
   over the normal/Wald interval specifically because it stays sane at
   small N and at rates near 0 or 100%, which several cells — especially
   the cold manipulation-check cell and any near-ceiling hot cells — are
   expected to hit). Report as `k/N (rate%, [lo%, hi%])` per cell, no
   further processing, for every cell in §3 and §4.
2. **Factor main effects** as a simple **risk difference** (invocation
   rate at one level minus rate at a comparison level, holding all other
   factors at their §1 fixed baseline), with a **Newcombe hybrid-score CI**
   on the difference (the standard small-sample-appropriate way to get a
   CI on a difference of two Wilson intervals without assuming normality).
   No logistic regression, **no formally-tested** interaction terms, no
   multiple-comparisons correction beyond eyeballing the CI widths — this
   is explicitly a scoping/characterization pass, not a hypothesis-test
   battery, and PLAN's own guidance is to keep this simple and honest
   rather than over-engineer statistics on an 11-cell, N=15 design. **One
   descriptive exception (no added cells, no formal test):** because the
   core block is a full A{3}×B{2} factorial, report the **six A×B cell
   rates as a 2×3 table** so the **overlap×specificity** interaction is
   *inspectable* (do specific descriptions help more when overlap is low?).
   This is presented as descriptive cell means only — no interaction CI is
   claimed, since at N=15 a difference-in-differences CI would exceed the
   ±30-pt bar in item 3. All interactions **involving C or D** are out of
   scope (the C/D sweeps are marginal-only; see §3 lock).
3. **Practical-significance bar, borrowed in spirit from the project's
   MDD discipline**: report a factor effect as "distinguishable" only if
   its Newcombe CI excludes 0 **and** the point estimate clears a coarse
   30-percentage-point gap. This number is a proposal, not derived from
   power analysis (the N wasn't chosen to hit a target power — it was
   chosen to fit the budget, and §6 is explicit about that ordering) —
   OPUS should treat 30 points as a starting anchor to accept, tighten,
   or loosen, not a statistically justified cutoff. Effects that don't
   clear this bar are reported as point estimates with CIs, explicitly
   labeled "not distinguishable from noise at this N," not silently
   dropped.
4. **D-necessity cells additionally report `solved` rate** alongside
   `invoked` rate, and the joint pattern (invoked-and-solved vs.
   invoked-but-wrong vs. not-invoked-and-unsolved vs. — the one
   theoretically impossible cell — not-invoked-but-somehow-solved, which
   would indicate the task wasn't actually need-engineered and should be
   flagged/discarded) is reported as a 2×2 count table per D-necessity
   cell rather than collapsed into a single number, mirroring the
   capability-vs-cash-out separation PLAN.md already uses for P10-C's
   gates.
5. **What is explicitly NOT claimed**: no p-values, no claim of
   statistical significance in the formal sense, no attempt to fit a
   model across all four factors jointly. If a later phase (P10.3 itself,
   after real data exists) wants a joint model, that is a new
   pre-registration with its own justification for the added complexity,
   not an extension silently bolted onto this one.
6. **Every reported number cites its run ID(s)** per the experiment-
   integrity rule — this pre-registration does not itself contain any
   results, only the plan for producing and reporting them.

## 7. Review resolutions (OPUS, 2026-07-11) + items left to the operator

**Resolved by this review (now binding in the sections above):**

1. **Invocation detector channel — RESOLVED.** The pin is Claude Code
   `2.0.14` (`infra/trial_logic.py`); every on-disk transcript's init tool
   list contains `SlashCommand` and **no** `Skill` tool, and skills surface
   under the init `slash_commands` field. Detector is **dual-channel,
   target-scoped**: channel S (`SlashCommand` tool_use whose
   `input.command` = `/<target>`, primary), channel K (`Skill`,
   forward-compat, expected 0), channel T (text `/<target>`); per-channel
   counts reported; `invoked` = S ∨ K (amended 2026-07-11 pre-data; T
   reported-only, see §2). Proof-of-one is
   P10.2's mandatory first step to pin the `input` shape (mechanism-notes
   §3, §2 outcome variable).
2. **Manipulation gate — RESOLVED: literal 80/20 is binding**, N=10/cell;
   Newcombe gap-CI reported only to classify a miss as separation-present
   vs. degenerate, mapped onto PLAN's "one redesign, then kill" (§4).
3. **Grid shape — RESOLVED: 11-cell reduced grid LOCKED.** It satisfies the
   P10-A gate (4 marginal factor effects). The overlap×specificity
   interaction is already free from the A×B factorial (reported
   descriptively, §6.2); C/D-involving interactions are deferred to a
   sequential follow-up, not bought at N=15 where their DiD CI would be
   uninterpretable (§3).
4. **Truncation reachability — RESOLVED.** At 20 skills with ~30-word
   descriptions on a 200k subject the listing (~1,000 tok) is under the
   ~2,000-tok budget → truncation UNREACHABLE as originally specified.
   Fixed: distractor descriptions mandated ≥~500 chars (toward the
   1,536-cap) with a binding `/context` build-time check that truncation
   engaged; and the cell is honestly re-scoped — zero invocation history in
   a fresh probe makes the documented drop-order (least-*invoked*-first)
   uncontrollable, so C-20-last tests count/dilution + truncation-presence,
   **not** a clean target-drop manipulation (§1 Factor C).

**Left to the operator (not blockers for committing this pre-registration):**

- Confirm N=10 (manipulation) / N=15 (grid), or bump to N=20/N=25-30, once
  P10.2's proof-of-one measures real per-trial cost (§5 shows headroom).
- Confirm **Sonnet 5** as the sole P10.3 reference subject (and its context
  window — 200k vs 1M — which sets the C-20 truncation arithmetic, §1), or
  approve a cheap Haiku secondary pass.
- Provide the P10.3 reference-model auth path (PLAN §P10.3 operator
  dependency); operator may veto reference-model API use entirely.
