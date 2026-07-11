# P10.1 — Progressive-disclosure mechanism notes

Authoritative reference for P10-A/B/C probe design. Compiled 2026-07-11 from
official Anthropic documentation, the Agent Skills open standard, and this
repo's own local evidence (v006-skill-library, A0 gate). Every claim is
tagged **[DOC]** (documented, cite given), **[OBS]** (observed in this
repo's own runs/config, run ID or file path given), or **[ASSUME]**
(inference not directly stated by either source — flagged as an open
question for the probe grid).

Sources (fetched 2026-07-11):
- Claude Code skills reference — `https://code.claude.com/docs/en/skills`
- Agent Skills overview (Claude Platform docs) —
  `https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview`
- Agent Skills authoring best practices —
  `https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices`
- Anthropic engineering: "Equipping agents for the real world with Agent
  Skills" — `https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills`
- Agent Skills open standard — `https://agentskills.io`
- Local: `.claude/skills/*/SKILL.md` (this repo's harness, read-only),
  `experiments/variants/v006-skill-library/` (manifest + 3 skills + CLAUDE.md
  pointer), `findings/FINDINGS.md` A0 entry (2026-07-10), `infra/export_trajectories.py`.

**Most authoritative single source for the mechanism**: the Claude Code
skills reference page (`code.claude.com/docs/en/skills`) — it is the
first-party spec for the exact product this project targets (Claude Code
harness), current as of the fetch date, and the only source with the
concrete truncation-budget numbers and the `Skill(name)` permission syntax
used below.

## 1. The three disclosure levels

**[DOC]** Skills load in three stages (Claude Platform docs, "How Skills
work"; Anthropic engineering post, same framing):

| Level | Content | When loaded | Token cost |
|---|---|---|---|
| 1 — Metadata | `name` + `description` from YAML frontmatter | Always, at session/turn startup, folded into the system-prompt-adjacent skill listing | ~100 tokens/skill (Platform docs table) |
| 2 — Instructions | SKILL.md body | When the skill is triggered (model-decided or user-invoked) | Under 5k tokens recommended; hard cap "under 500 lines" (best-practices) |
| 3+ — Resources | Bundled files (`reference.md`, `scripts/*.py`, …) referenced from SKILL.md | Read on demand via the model's own tool calls (bash/Read), only if the body's own logic reaches for them | Effectively unbounded — zero cost until read |

**[DOC]** Mechanically, in Claude Code specifically: "Claude Code watches
skill directories... At startup, the `name` and `description` from all
Skills' YAML frontmatter are loaded into the system prompt" (best-practices
"Runtime environment" section); when triggered, "Claude uses bash Read
tools to access SKILL.md and other files from the filesystem" (same
section, and Platform-docs overview "How Claude accesses Skill content").
In Claude Code proper (not the raw code-execution container the Platform
docs describe for claude.ai/API), the mechanism for reading SKILL.md is the
harness's own file-read machinery, not necessarily a bash subprocess — see
§3 for the invocation channel actually exposed to the model.

## 2. The frontmatter contract

**[DOC]** Two fields, both governed by hard validation rules (best-practices
"YAML frontmatter requirements", identical text on the overview page):

- `name` — optional in Claude Code (defaults to the directory/file name;
  see below), but where present: **max 64 characters**, lowercase
  letters/numbers/hyphens only, no XML tags, cannot contain the reserved
  words "anthropic" or "claude".
- `description` — **max 1024 characters** in the generic Agent Skills spec;
  **must be non-empty**; no XML tags; "should describe what the Skill does
  and when to use it."

**[DOC] Claude-Code-specific extension** (code.claude.com/docs/en/skills,
"Frontmatter reference" table): the *listing* entry built from
`description` + optional `when_to_use` is truncated at **1,536 characters**
combined (`skillListingMaxDescChars`, configurable) — "Put the key use case
first." `when_to_use` is an additional Claude-Code-only field for
"trigger phrases or example requests," appended to `description` and
counting toward the same 1,536-char cap.

**[DOC] What makes a description "trigger well"** (best-practices "Writing
effective descriptions", direct quotes):
- **Third person, always.** "Processes Excel files and generates reports" —
  not "I can help you..." or "You can use this to...". Rationale given: the
  description is injected into the system prompt, and a shifted
  point-of-view "can cause discovery problems."
- **Specific, with key terms and explicit trigger conditions** — "include
  both what the Skill does and when to use it." Worked example: *"Extract
  text and tables from PDF files, fill forms, merge documents. Use when
  working with PDF files or when the user mentions PDFs, forms, or document
  extraction."* Anti-examples explicitly called out as too vague: "Helps
  with documents", "Processes data", "Does stuff with files."
- Selection is described as competitive: "Claude uses [the description] to
  choose the right Skill from potentially 100+ available Skills" — i.e. the
  description is doing discriminative work against siblings, not just
  describing the skill in isolation.
- **[DOC]** Naming convention (secondary, affects how skills are discussed
  and organized but not shown to matter for invocation likelihood):
  gerund form (`processing-pdfs`) is recommended; vague/generic/reserved
  names discouraged.

## 3. How invocation happens mechanically — the detector

**[DOC] Current Claude Code (code.claude.com/docs/en/skills, "Restrict
Claude's skill access")**: model-driven invocation goes through a tool
literally named **`Skill`**. Permission syntax is `Skill(name)` /
`Skill(name *)`; denying the bare `Skill` tool disables all model-invoked
skills. `disable-model-invocation: true` in frontmatter removes a skill
from the model's candidate set entirely (still user-invocable via
`/name`). A skill also surfaces as a `/name` slash command; typing it is a
*user*-initiated invocation, not the channel this project's probes care
about (the probes test whether the *model* decides to invoke, not whether a
human does).

**[DOC] What a transcript shows.** This project's harness records Claude
Code / Agent-SDK transcripts as Anthropic Messages-API content blocks
(confirmed locally: `infra/export_trajectories.py` module docstring —
"Canonical order is thinking -> text -> tool_use," blocks carry
`id`/`name`/`input` verbatim). A skill invocation is therefore, in the
current documented mechanism, an assistant-turn content block with
`type: "tool_use"` and `name: "Skill"` (input carries the skill name and
any arguments). **This is the P10.2 detector**: scan each trial's assistant
turns for a `tool_use` block whose `name == "Skill"` (or `SlashCommand` —
see the version caveat immediately below); a trial "invokes" if at least
one such block appears before the trial ends.

**[OBS] Version caveat — must be verified empirically before P10.2 locks
the detector.** This repo's own A0 manipulation check (run
`20260709T235258` + `20260710T000737`, FINDINGS 2026-07-10) recorded
skills surfacing in the headless worker's **init event** under a field
named `slash_commands`, and described the invocation channel as the
**"SlashCommand tool"** — not "Skill." That run predates this session and
may reflect an earlier Claude Code / Agent SDK release where skills were
still modeled as slash commands proper (the current docs note: "Custom
commands have been merged into skills" — implying a schema change
happened at some point). **Action for P10.2**: before trusting the
`name == "Skill"` filter, run one proof-of-one trial against whatever
Claude Code / Agent SDK version this harness's runner actually pins, and
confirm the literal tool name in the recorded `tool_use` block. Do not
assume the current docs' name is what the pinned CLI version emits.

**[DOC] Two other invocation-adjacent signals worth logging alongside the
detector**, both from the current docs:
- **Passive listing presence**: every skill's name+description is in the
  system-prompt-adjacent listing regardless of use — this is not
  invocation and must not be conflated with it (this is exactly the
  distinction the v006 postmortem drew, see §5).
- **Re-invocation dedup**: "When Claude re-invokes a skill whose rendered
  content is identical to the copy already in context, Claude Code adds a
  short note that the skill is already loaded rather than a second copy"
  (skills reference, "Skill content lifecycle"). For single-turn short
  probes this should never trigger, but if a probe task allows multiple
  turns, a second `Skill` tool_use call on the same skill still counts as
  invocation for the detector even if the harness elides the content.

## 4. Documented ranking / truncation under many skills

**[DOC]** This is the one place the docs give a concrete, load-bearing
mechanism for the "skill count" factor (code.claude.com/docs/en/skills,
"Skill descriptions are cut short"):
- The skill listing always contains **every skill's name**.
- The **character budget for the listing scales at 1% of the model's
  context window** (`skillListingBudgetFraction`, default fraction,
  configurable). `/doctor` reports an estimate of the listing's cost;
  `/context` reports the actual post-budget size.
- **When the listing overflows the budget, Claude Code drops descriptions
  starting with the skills you (the *user*) invoke least** — i.e. the
  truncation criterion is **historical invocation frequency across
  sessions**, not list position, not alphabetical order, and not
  recency of installation. A dropped-description skill still appears
  by name (so the model knows it exists) but loses the trigger text that
  would let it match a task.
- Per-entry cap is separate and always active regardless of overall budget
  pressure: the 1,536-char combined `description`+`when_to_use` cap (§2).
- Troubleshooting guidance for "skill triggers too often" is to *shorten
  the description*; for "not triggering," to check the description's
  keyword coverage and confirm the skill is even listed (`What skills are
  available?`).

**[DOC]** No mechanism is documented for *list position* (first vs. last in
the listing) mattering independently of the frequency-based drop rule
above. The docs describe order effects only for the truncation criterion
(usage-frequency), never for primacy/recency within an already-fit
listing.

## 5. Local evidence

**[OBS] This harness's own `.claude/skills/*/SKILL.md`** (read-only,
inspected, not edited) all use the documented optional-field surface
correctly and consistently: `description` in third person with an explicit
"Use when..." clause and quoted trigger phrases (e.g.
`classify-failures/SKILL.md`: *"Use after any eval run when asked to
'classify failures', 'analyze the failed runs', 'what went wrong in run
X', or before proposing a scaffold mutation."*), plus Claude-Code-specific
fields not covered by the generic spec — `argument-hint`, `arguments`,
`allowed-tools`, and (on `classify-failures`) `context: fork`. These map
directly onto the frontmatter table in §2/§3: `context: fork` runs the
skill in an isolated subagent per the docs' "Run skills in a subagent"
section — worth noting because a forked skill's invocation still shows up
as a `Skill` tool_use in the parent transcript even though its execution
happens elsewhere.

**[OBS] v006-skill-library** (`experiments/variants/v006-skill-library/`,
status: rejected) is the closest prior art to P10's probe grid in this
repo. Three concrete skills (`reproduce-before-editing`,
`run-the-verify-contract`, `localize-by-symbol`) were written as genuinely
necessity-adjacent procedures (distilled from Ornith's own
verifier-passing trajectories, not invented), each with a description
following the documented pattern (third person, explicit "Use
before/when/to..." clause), plus a one-line CLAUDE.md pointer naming one
skill explicitly (`/reproduce-before-editing`). Manifest + FINDINGS record
the outcome precisely:
- **0/12 trials invoked any skill** via the (then-named) SlashCommand
  channel, despite skills surfacing correctly in the init event and the
  system prompt naming one by its invocable form.
- **A measurable passive effect existed anyway**: skill descriptions
  sitting in context shifted behavior on the same 4 anchor tasks — mean
  reproduction-script runs 1.5 vs 0.5 (3x), mean verify runs 1.8 vs 0.4
  (4.5x) — without any `Skill`/`SlashCommand` tool_use ever appearing.
  This is direct local evidence that Level-1 metadata (description text)
  can influence behavior even with zero invocations at Level 2 — a
  mechanism the official docs don't discuss (they treat Level 1 purely as
  a selection signal, not as freestanding context that shapes generation).
- The postmortem explicitly reasons this is a property of *this model*
  (Ornith-35B), not of the mechanism: "Ornith's operative behaviors were
  REWARDED in [RL on a self-authored persistent scaffold], not prompted."
  This is exactly why P10-A's reference-model characterization matters —
  the mechanism-level facts in §1-4 say nothing about invocation
  *propensity*, only about what's visible and how it's triggered.

## 6. Documented fact vs. observed behavior vs. assumption — summary table

| Claim | Status |
|---|---|
| Three-level progressive disclosure (metadata always, body on trigger, resources on demand) | **[DOC]** — Platform docs, engineering post, agentskills.io all agree verbatim |
| `name`≤64 chars, `description`≤1024 chars (spec) / 1536 chars combined listing cap (Claude Code) | **[DOC]** |
| Description should be third-person, specific, state both what+when | **[DOC]** — explicit authoring guidance with good/bad examples |
| Invocation tool is literally named `Skill` in current Claude Code | **[DOC]** for the current doc revision; **[OBS] contradicted** by this repo's own A0 transcripts (`SlashCommand`, `slash_commands` init field) from an earlier session/version |
| Truncation under many skills drops least-*invoked* skills' descriptions first, budget = 1% of context window | **[DOC]** — precise, load-bearing mechanism for the count factor |
| List *position* (first/last) independently affects selection or truncation | **not documented either way** — **[ASSUME]** open question |
| Passive behavioral shift from Level-1 metadata alone, without invocation | **[OBS]** — this repo's v006 A0 result; not discussed in official docs at all |
| Whether "necessity" (task unsolvable without skill body) raises invocation odds vs "convenience" | **not documented** — **[ASSUME]** |
| Whether small/cheap models (Haiku-tier) invoke skills at different rates than Sonnet/Opus | **[DOC]** only as a *testing* recommendation ("test with Haiku, Sonnet, Opus... what works for Opus might need more detail for Haiku") — no quantified propensity data given |
| Description length vs. specificity as independent knobs | Not separated in the docs' guidance (concise + specific are both urged, with no stated tradeoff) — **[ASSUME]** these can be manipulated independently for the grid |

## 7. Open questions the probe grid must answer empirically

These are exactly the gaps the sections above could not close from
documentation or prior local runs, and they map directly onto the four
PLAN.md factors:

1. **List position.** Given many skills at the same description quality,
   does position in the listing (first vs. last presented, independent of
   the documented frequency-based truncation) measurably change invocation
   odds? The docs give no mechanism for this; if the probe grid finds an
   effect, it would be a genuinely new finding, not confirmation of a
   documented behavior.
2. **Semantic-overlap threshold.** How much lexical/semantic distance
   between task text and description survives before invocation drops off
   — is it closer to a hard keyword-match gate or a graded similarity
   response? Not addressed by any source; the best-practices guidance
   ("include specific triggers... the user mentions X") implies keyword
   matching matters but gives no quantification.
3. **Necessity vs. convenience.** Do models invoke more reliably when the
   skill body is load-bearing (task literally unsolvable without it) vs.
   merely helpful? No source discusses this distinction; it interacts
   with the "convenience" framing implicit in the docs' own examples,
   which are all convenience-type (PDF extraction is rarely
   unsolvable-without-the-skill).
4. **Exact tool name / transcript shape for this project's pinned
   harness version.** §3's version caveat — must be confirmed on a
   proof-of-one trial before the P10.2 detector is finalized, given the
   documented-vs-observed discrepancy already on record in this repo.
5. **Count-vs-truncation interaction at small N.** The documented
   truncation mechanism only engages once the listing exceeds ~1% of
   context window — for a small number of installed skills (the likely
   regime for short probe tasks) is truncation simply inactive, making
   "skill count" purely a search-space/attention-dilution factor rather
   than a description-completeness factor? This determines whether the
   count factor's mechanism of action is "shorter description" or "more
   candidates to discriminate among."
6. **Model-tier propensity gradient.** Given the docs' own recommendation
   to test skills across Haiku/Sonnet/Opus separately, does invocation
   rate vary systematically by model tier at fixed description quality?
   This bears directly on P10-B (predicting where Ornith's near-zero
   result sits relative to the reference-model floor/ceiling).
