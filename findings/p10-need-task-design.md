# P10.5 phase 1 — need-engineered task families (DESIGN — for orchestrator review)

Status: **DRAFT for review**. Author: OPUS corpus role, 2026-07-11
(PLAN.md §P10 phases, P10.5). Companion to the SONNET generator work.
Nothing here has been run for score; three exemplars exist on disk under
`experiments/probes/need-exemplars/` with verifiers tested both ways (§6).

**What this is.** A design for ~30 hermetic micro-tasks in which a
**workspace skill carries load-bearing information absent from the task
text**, such that (a) solving without consulting the skill is near-impossible,
(b) solving with it is easy for a competent model, and (c) a deterministic
VERIFY command decides success. These are the P10.5 corpus that gates P10.6's
two separated measures: **invocation rate on a held-out split** (capability)
and **paired solve delta** (cash-out). This design generalizes Factor D of the
probe grid (`p10-probe-grid-draft.md` §1 Factor D — the ticket-ID cell) from a
single cell into a family battery, under the corpus-curator admission bar
(hermeticity / determinism / provenance / no-oracle-leakage).

**Two decisions are flagged for orchestrator sign-off** and called out inline:
the **held-out split rule** (§3) and the **consultation metric** (§4). The
second refines the PLAN's P10-C gate wording and should be signed off before
P10.6 scores anything.

---

## 0. Structural invariants (binding)

- These task workspaces are **DATA** under `experiments/probes/`, exactly like
  the probe-grid workspaces and `experiments/variants/*/claude-config/`. Each
  task ships its own `.claude/skills/<skill>/SKILL.md` — the **subject** skill
  under test — which is **never** this repo's own `.claude/` and never copied
  into it. The one structural invariant (CLAUDE.md) holds.
- `tasks/holdout/`, `tasks/dev/`, and this repo's `.claude/` are not touched by
  this work. Committing the exemplars is the caller's decision; this session
  leaves them uncommitted for review.
- **The hidden-verify contract (load-bearing anti-gaming invariant).** For
  every task, `verify.sh` and `SOLUTION.md` are **harness-side only** — never
  materialized into the worker's workspace, exactly the contract `hidden_tests`
  already use (tasks/schema.md "Eval-verdict contract"). The worker sees only
  `workspace/` + `.claude/` + `task.md`. This is what makes the verifier
  non-leaky: the check logic (and any embedded grammar or expected marker)
  cannot be read off a file the worker never receives. Materialization layout:
  `workspace/` → worker CWD; `.claude/` → worker config; `task.md` → prompt;
  `verify.sh` runs at grade time against the returned `workspace/`.
- Hermeticity: no network, `python3` stdlib + bash only in every verifier,
  fixed container image (reuse the pinned probe/runner image; no build step).

---

## 1. Task families

Six families, each defined by **the kind of fact the skill carries**. Every
family shares the same skeleton — `task.md` states a goal but omits the
load-bearing fact; the skill supplies it; `verify.sh` checks it deterministically
— and differs in *what* is withheld and *how* the verifier stays non-guessable.

### F1 — magic-constant (opaque literal value)

- **Template.** Workspace has a stub function/config that must return or emit a
  fixed project constant (sync token, license key, canonical port, feature-flag
  salt, region code). `task.md`: "return the project's canonical `<thing>`;
  it's a fixed constant, do not invent it."
- **Skill carries.** The literal value, e.g. a 12-hex sync token
  `9f3ac1d70b2e`, plus a one-line "return it exactly as shown" instruction.
- **Task omits.** The value itself. Task names the *concept* ("sync token") but
  never the bytes.
- **VERIFY design.** Import the artifact, read the emitted value, compare a
  **salted SHA-256 digest** to a hard-coded expected digest. The literal never
  appears in the verifier.
- **Unguessable / entropy.** A 12-hex constant spans ~2^48 ≈ **2.8×10^14**; an
  8-hex constant spans ~2^32 ≈ **4.3×10^9**. One blind guess passes with
  probability ~1/space. Use ≥12-hex so that even a leaked digest is
  brute-force-infeasible; 8-hex is acceptable only because the digest is
  harness-side and never exposed.
- **Brute-force failure mode.** The worker has no in-loop access to the
  verifier (harness-side), so there is no digest to grind against and no
  pass/fail oracle to binary-search. Even with the digest, 2^48 salted SHA-256
  preimages is infeasible in a task budget. A key-stretched digest (100k rounds)
  closes the residual 8-hex risk if a shorter constant is ever required.

### F2 — bespoke-dsl (grammar / format rules)

- **Template.** Author a config/manifest file in a project-native mini-format
  (service manifest, routing table, policy file). `task.md`: "write
  `<file>` declaring `<items>` in the project's native format."
- **Skill carries.** The grammar: required header sigil + version, per-line
  record syntax, field ordering, padding rules, and a **computed checksum/footer
  rule** (e.g. `(sum of name lengths) mod 97`, zero-padded).
- **Task omits.** All grammar. It gives only the semantic payload (which items,
  in which order).
- **VERIFY design.** A stdlib parser embedded in `verify.sh` that checks the
  header exactly, each record against the syntax, ordering, and recomputes the
  checksum. Exit 0 only if every rule passes.
- **Unguessable / entropy.** Combinatorial across independent arbitrary choices:
  header sigil (unbounded string), record prefix, pad width, and — decisively —
  the **checksum function** (which of unboundedly many arithmetic rules). The
  checksum alone makes a from-scratch reconstruction have negligible
  probability; even a correctly-structured guess fails on the footer.
- **Brute-force failure mode.** No oracle loop. The checksum is a
  content-dependent computation the model cannot reproduce without the rule.
- **DSL family caveat (honest tradeoff).** Unlike F1, this verifier **must embed
  the grammar** to check it generatively, so it **cannot be salted/hashed**. Its
  non-leakability rests **entirely** on the hidden-verify contract (§0). Flagged
  so no reviewer assumes the F1 digest trick generalizes here.

### F3 — registration-convention (arbitrary cross-file procedure)

- **Template.** Add a feature (a command, a plugin, a route handler). The
  project convention requires **registering** it in a manifest with a specific
  format **and** naming the implementing symbol by a specific rule. `task.md`:
  "add a `<name>` command so the app dispatches it."
- **Skill carries.** The convention: e.g. "register in `registry.txt` as
  `cmd:<name>:<slot>` where `<slot>` is the 1-based line index of `<name>` in
  `COMMANDS.md`; the handler **must** be named `_handle__<name>` (double
  underscore)."
- **Task omits.** Both the registry format and the symbol-naming rule. The
  natural guess (single underscore, no registry) is wrong.
- **VERIFY design.** Check the registry line matches the exact format with the
  correctly computed slot, and that the handler symbol with the double-underscore
  name exists and is dispatchable (import + call).
- **Unguessable / entropy.** The double-underscore sigil and the slot-derivation
  rule are arbitrary; the *natural* convention (single underscore, append-order)
  is a plausible attractor that is **wrong**, so a non-consulting model is
  actively lured to the wrong answer rather than merely missing.
- **Brute-force failure mode.** Two independent arbitrary bits (registry format
  × symbol rule) must both be right; guessing one does not reveal the other, and
  there is no oracle.

### F4 — nonstandard-invocation (bespoke toolchain contract)

- **Template.** Make a module pass the project's **bespoke checker/runner** — a
  custom loader/validator invoked by a non-obvious command. `task.md`: "make
  `plugin.py` load under the project's plugin checker."
- **Skill carries.** The command *and* the contract the code must satisfy to
  pass it: e.g. "the loader requires a module-level `PLUGIN_API = 7` (this
  project pins API 7) and an `entrypoint(ctx)` function; run
  `python3 tools/plugcheck.py <file>`."
- **Task omits.** The API-version integer, the required entrypoint name, and (to
  the worker) the runner's internals — the runner is part of the verifier
  (harness-side), so the worker cannot read the contract off it.
- **VERIFY design.** Run the bespoke checker; it passes only when the module
  exposes the correct API sentinel and entrypoint signature.
- **Unguessable / entropy.** The API integer is an arbitrary small constant
  (guess space large enough that a blind pick is improbable, and *wrong values
  are silently rejected*), and the entrypoint name is one of unboundedly many.
  Combined, near-zero blind pass rate.
- **Brute-force failure mode.** The checker is harness-side; the worker cannot
  enumerate API values against it. Distinct from F1 in that the fact is a
  *contract shape* (a named symbol + a sentinel) rather than a single emitted
  literal.

### F5 — broken-default-workaround (deliberate trap + non-obvious remedy)

- **Template.** The default build/run path is **deliberately sabotaged** (the
  obvious module raises; the default make target is poisoned). `task.md`: "the
  default path is broken in this build; fix it using the project's *supported
  workaround* — a naive fix that only returns the right value is not enough."
- **Skill carries.** The two-part remedy: switch to the sanctioned path **and**
  emit a specific non-obvious marker (a `.procrc`/pragma/env token) whose exact
  filename and content are project convention.
- **Task omits.** The marker. The *broken-ness* is visible; the *remedy* is not.
- **VERIFY design.** The verifier is the "CI gate": it (1) checks the marker
  (salted-hashed content, harness-side) and (2) checks the artifact runs
  correctly. The marker check gates first, so a correct-output-but-no-marker fix
  fails.
- **Unguessable / entropy.** A competent model *can* discover the sanctioned
  code path from the workspace and get the right output the natural way — that
  is the designed trap: it produces a plausibly-complete solution that the marker
  gate rejects. The marker (filename + exact line) is an arbitrary-identifier
  space with no in-workspace hint and no feedback loop.
- **Brute-force failure mode.** The verifier is harness-side, so the model
  believes it is done and never learns the marker is missing — there is nothing
  to grind against. This is the family most robust to "reason harder," because
  reasoning *converges on the wrong-but-plausible* answer.

### F6 — lookup-table (many-entry arbitrary mapping)

- **Template.** Implement a function honoring an arbitrary N-entry mapping
  (status-code → canonical message, error-class → retry budget, region →
  shard-id). `task.md`: "implement `canonical_message(code)` for the project's
  status codes."
- **Skill carries.** The full mapping table (e.g. 6–10 arbitrary pairs), often
  with a documented default/fallback rule.
- **Task omits.** Every mapping value. The keys may be conventional; the values
  are not.
- **VERIFY design.** Call the function on **several** keys (including the
  fallback) and compare each output. Multiple independent checks defeat partial
  guessing.
- **Unguessable / entropy.** Each entry is independently arbitrary; checking k
  entries multiplies the per-entry improbability. Distinct from F1 (one
  constant) in that it is a *structured multi-value* fact, and from F6-vs-F2 in
  that there is no grammar, only opaque values.
- **Brute-force failure mode.** Salted-digest per-entry comparison keeps values
  out of the verifier; the joint space of k arbitrary values is unguessable and
  the harness-side verifier gives no per-entry feedback.

**Family distinctness (why six, not fewer).** The axis is *fact kind*: F1 opaque
scalar · F2 grammar · F3 cross-file procedure · F4 toolchain contract · F5
deliberate-trap remedy · F6 structured mapping. F3 and F5 each add an
**attractor** (a plausible-but-wrong natural answer) that pure-improbability
families (F1/F6) lack — valuable because they separate "model that guessed
right" from "model that consulted."

---

## 2. What "load-bearing + non-gameable" requires (design rules, applied per family)

Every instance must satisfy, and the generator (§3) asserts, all of:

1. **Load-bearing:** the withheld fact has entropy ≥ ~30 bits *or* is an
   attractor-trap (F3/F5) where the natural answer is wrong. No fact recoverable
   from the workspace, task text, or common convention.
2. **Solvable-with:** the skill body states the fact unambiguously; the act is a
   ≤10-line edit / one file. A competent model that has read the skill finishes
   in one or two turns.
3. **Deterministic verify:** exit code only, no LLM in the verdict path, stdlib
   only, offline; passes the 3+3 determinism check (3× on solution → 0, 3× on
   the stub/plausible-attempt → nonzero).
4. **No oracle leakage:** verifier + solution are harness-side (§0); wherever
   the fact is a *value* (F1/F5/F6) the verifier stores only a salted digest;
   wherever it is a *grammar/contract* (F2/F4) the verifier is inherently
   revealing and relies on the hidden-verify contract alone (flagged per family).
5. **No convention leakage:** the load-bearing fact must be arbitrary, not the
   community-default guess. F3/F5 go further and make the default guess *wrong*.

---

## 3. Generator plan (~30 instances)

### 3.1 Template → instance

Each family is a **template** parameterized by a seed. A string-seeded RNG
(same discipline as the probe harness's trial-indexed RNG) draws:

- **Names.** Function/file/skill/service/command names from disjoint word pools
  (so no two instances share identifiers that could let a model cache an answer).
- **Constants.** F1/F6 values are drawn fresh (12-hex tokens, mapping values);
  F2 checksum inputs vary with the drawn item names, so the correct checksum
  differs per instance; F4 API sentinels and entrypoint names are drawn; F5
  marker filenames + lines are drawn.
- **Layout.** Shallow vs. nested package, file count, decoy files (unrelated
  source that does not reveal the fact) to vary surface without adding signal.

The generator emits, per instance, the full dir (`workspace/`, `.claude/skills/…`,
`task.md`, `verify.sh`, `SOLUTION.md`) and **re-derives the verifier's expected
digest/checksum from the drawn seed** so the verifier and skill can never drift.
Post-generation, the generator runs each verifier 3× on the auto-produced
solution (must be 0) and 3× on the stub + one canned plausible-attempt (must be
nonzero) — the corpus-curator 3+3 determinism gate, automated.

### 3.2 Counts

| Family | Instances | Notes |
|---|---|---|
| F1 magic-constant | 5 | different constant types (token/port/key/salt/region) |
| F2 bespoke-dsl | 5 | manifest / routing-table / policy variants, distinct checksum rules |
| F3 registration-convention | 5 | command / plugin / route registries |
| F4 nonstandard-invocation | 5 | loader / validator / linter contracts |
| F5 broken-default-workaround | 5 | import-trap / make-target-trap / env-trap |
| F6 lookup-table | 5 | 6–10 entry mappings, distinct domains |
| **Total** | **30** | 5 per family; expandable to 6–8/family (→ 36–48) if P10.6 wants more power |

### 3.3 Split rule — FLAGGED DECISION (held-out = whole FAMILIES)

**Proposal: family-level holdout.** Reserve **2 whole families** (e.g. F4 +
F6, ~10 instances) as the held-out split; train the consult-then-act SFT only
on the other 4 families' instances (~20). Rationale:

- The P10-C capability gate is literally **"invocation rate on HELD-OUT
  need-engineered probes ≥50%"** (PLAN.md). The scientific question is
  *transfer*: did SFT instill a **general** consult-then-act disposition, or did
  it memorize "these task shapes need a skill"? **Held-out families test
  transfer**; held-out *instances* of trained families test only memorization
  (the model has seen the shape, just not the seed).
- Family holdout is the stronger, more honest gate and matches the spirit of the
  project's contamination discipline (improvement that doesn't generalize across
  the split is not progress).
- **Cost of family holdout:** fewer distinct shapes in train (4 families). Mitigate
  by keeping per-family instance counts up (≥5 train families' instances) and, if
  the orchestrator wants both signals, report a **secondary** held-out-instance
  measure on the trained families as a memorization-vs-transfer contrast.

**Recommendation to flag:** primary gate on **held-out families** (transfer);
optionally also report held-out-instances-of-trained-families as the
memorization baseline. Which 2 families to hold out is an orchestrator choice —
suggest holding out one improbability family (F6) + one attractor-trap family
(F4 or F5) so the held-out split exercises both "model that never guesses" and
"model lured to a wrong default." **Orchestrator: confirm family-level holdout
and the specific held-out families.**

### 3.4 Hermeticity

No network in any verify; `python3` stdlib + bash only (verified in the three
exemplars); fixed container image (reuse the pinned runner image — no per-task
build). Each instance is self-contained and reproducible from `(family, seed)`.

---

## 4. Anti-gaming audit + the consultation metric — FLAGGED DECISION

### 4.1 Per-family: can a model pass WITHOUT consulting?

| Family | Non-consult pass path | Closed by |
|---|---|---|
| F1 magic-constant | guess the constant | ~2^48 space; harness-side salted digest → no oracle |
| F2 bespoke-dsl | reconstruct the grammar+checksum | checksum is a content-dependent arbitrary rule; harness-side verifier |
| F3 registration | use natural convention (single `_`, append order) | natural guess is *wrong* by construction; two independent arbitrary bits |
| F4 nonstandard-invocation | guess API sentinel + entrypoint name | wrong sentinels silently rejected; harness-side checker → no enumeration |
| F5 broken-default | fix output the natural way, skip the marker | marker gate is harness-side + a plausible wrong answer → no feedback loop |
| F6 lookup-table | guess mapping values | k independent arbitrary values; per-entry salted digests → no oracle |

The common closure is the **hidden-verify contract** (§0): the worker never
receives the check logic, so no family offers an in-loop pass/fail oracle to
binary-search, and no expected value/grammar can be lifted from a file the
worker sees. Families that *could* leak via an inherently-revealing verifier
(F2 grammar, F4 contract) are flagged as relying on the contract alone; the
value families (F1/F5/F6) additionally salt-hash so even a leaked verifier
reveals nothing usable.

### 4.2 What counts as "consultation" — reconciliation (FLAGGED for sign-off)

There is a genuine tension the PLAN's gate wording does not yet resolve, and it
must be settled before P10.6 scores:

- The **SFT** teaches *consult-then-act*: read the skill's content, then act.
- The **probe-grid detector** (`infra/probe_harness.py:detect_invocation`,
  `p10-mechanism-notes.md` §3) counts **only tool-call invocation** —
  channel S (`SlashCommand` → `/<skill>`) ∨ channel K (`Skill`). It does **not**
  count a file read of the SKILL.md.
- But in **these tasks**, the SKILL.md sits in the workspace's `.claude/skills/`.
  A model can obtain the load-bearing fact by **reading the file directly**
  (`Read`/`cat .claude/skills/<skill>/SKILL.md`) without ever emitting a
  `SlashCommand`. That read **is** consultation — it is the model choosing to go
  to the skill for the fact — and it is exactly the disposition the SFT instills.
  Counting only tool-call invocation would **undercount** success here and
  misreport a model that reliably consults-via-read as a failure.

**Proposed honest metric (refines PLAN P10-C's "invocation" gate for the
need-task split):**

- **Primary — `consulted ∈ {0,1}` = any skill-content access:** a
  `SlashCommand`/`Skill` **tool-call invocation** of the target skill **OR** a
  file **read** of the target `SKILL.md` path (or any bundled resource under
  that skill dir), detectable from the transcript (a `Read` tool_use whose path
  is under `.claude/skills/<target>/`, or a `Bash` `cat`/`sed`/`head` of that
  path). This is the capability gate for the need-task split: ≥50% held-out.
- **Secondary — channel breakdown:** report the split of `consulted` into
  **invocation-channel** (S∨K) vs **read-channel** (file read) vs **both**. This
  preserves the probe-grid's stricter tool-call signal as a sub-measure and lets
  the orchestrator see whether SFT produced *slash-command* invocation or merely
  *file reading* — a meaningful distinction for the P10-B/P10-A comparison, whose
  detector is tool-call-only.
- **Cash-out (unchanged):** paired **solve delta** (`verify` exit 0) on the
  need tasks, per the standing ≥+5-per-40 discipline scaled to the set size.
  This is the P10-C(2) gate and is independent of *how* the fact was obtained.

**Why this is the honest call.** The load-bearing fact lives only in the skill;
`verify` cannot be passed without obtaining it; so **any** pass is prima facie
evidence of consultation via *some* channel, and a solve without any detectable
skill-content access would itself be a red flag that the task leaked (to be
investigated, per the "not-invoked-but-somehow-solved → discard" logic already
in the probe grid §6.4). Making the primary metric "any skill-content access"
aligns the gate with what the SFT actually teaches and with what the tasks
actually require, while the secondary breakdown keeps the tool-call-only signal
comparable to P10-A/B.

**Orchestrator: sign off on (i) primary = any skill-content access (invocation
OR skill-file read), (ii) secondary = channel breakdown.** This is a refinement
of the PLAN's P10-C(1) wording ("invocation rate"), not a contradiction — flag
it in FINDINGS as the P10.5 metric reconciliation, mirroring the P10.2 pre-data
amendment that scoped channel T out of `invoked`.

### 4.3 Detector implementation note

The read-channel detector is a small extension of the existing
`detect_invocation`: in addition to scanning `tool_use` for `SlashCommand`/`Skill`,
scan for `Read` tool_use blocks and `Bash` blocks whose argument path resolves
under `.claude/skills/<target>/`. Both are already in the transcript stream-json
the probe harness parses; no new instrumentation. (Implementation lands in P10.6's
scoring, not here.)

---

## 5. Cost / size math (P10.6 ≤ $30 GPU)

**Per-task token profile.** These are deliberately small: `task.md` ~80–150
tokens; the skill listing (Level-1 metadata) ~60–120 tokens; on consult, the
SKILL.md body ~150–400 tokens; the act is a ≤10-line edit. A full trajectory
(system + task + one consult + one edit + verify round) is well under a
SWE-bench trajectory. Estimate **~2.5–5k tokens in / ~0.6–1.2k out per trial**,
comparable to the probe-grid per-trial figure.

**P10.6 is Ornith via `serve()` on Modal — billed by warm H100-seconds, not
per-token** (same structure as `p10-probe-grid-draft.md` §5b: ~$3.95/H100-hr,
~900 tok/s aggregate under concurrency). Per-trial token cost is a rounding
error against warm-container wall-clock. The binding constraint is warm
wall-clock across the SFT eval passes.

**Trials.** SFT gates per PLAN P10-C: (1) invocation/consultation on held-out
need tasks, (2) paired solve delta on need tasks, (3) dev-40 no-regression.
With ≤2 data recipes and, say, 5 trials/task for stable per-task rates:

- Held-out need split (2 families ≈ 10 tasks) × 5 = **50 trials** per checkpoint.
- Train-split solve delta (≈ 20 tasks) × 5 = **100 trials** per checkpoint,
  paired pre/post → ×2 = 200.
- Dev-40 no-regression is already-budgeted P10.6 work (the standard dev sweep),
  not new need-task cost.
- Two recipes → roughly ×2 on the need-task passes: ~**(50 + 200) × 2 ≈ 500
  need-task trials** total, plus the pre-SFT baseline pass.

**Cost.** At ~900 tok/s and ~4k tok/trial, 500 short trials ≈ 500 × ~4.4 s ≈
~37 min of *generation* time; with concurrency (the harness runs fan-out) and
cold-boot overhead, budget **~1.5–2.5 warm H100-hours ≈ $6–10** for the entire
need-task eval battery across two recipes and pre/post checkpoints. That sits
comfortably inside P10.6's **≤$30** cap, which is dominated by the **SFT
training runs** themselves (the LoRA/train pipeline), not by this eval corpus.
The corpus adds single-digit dollars.

**Generation cost:** $0 GPU — the generator is a stdlib Python script run
locally, producing all 30 dirs + running the 3+3 determinism gate offline.

---

## 6. Exemplars (built + tested)

Three exemplars, one each from three distinct families, live under
`experiments/probes/need-exemplars/`:

- `magic-constant-sync-token/` (F1)
- `bespoke-dsl-service-manifest/` (F2)
- `broken-default-workaround-legacy-shim/` (F5)

Each holds `workspace/` (worker-visible), `.claude/skills/<skill>/SKILL.md`
(the subject skill), `task.md`, `verify.sh` (harness-side, stdlib/bash),
and `SOLUTION.md` (harness-side, the intended consult-then-act path). Every
`verify.sh` was tested both ways — nonzero on an unconsulted plausible attempt,
0 on the intended solution:

```
F1 magic-constant : stub → exit 1 ; guess "deadbeefcafe" → exit 1 ; solution → PASS (0)
F2 bespoke-dsl    : no file → exit 1 ; plain newline list → exit 1 ; solution → PASS (0)
F5 broken-default : untouched → exit 1 ; compat-fix w/o .procrc → exit 1 ; solution → PASS (0)
```

The F5 negative case is the important one: a *plausible, output-correct* fix
(switch to `dataproc.compat`, `run()` returns `[2,4,6]`) still fails because the
skill-only `.procrc` marker is absent — demonstrating the attractor-trap keeps
the task load-bearing even against a competent non-consulting model.

---

## 7. Open items for the orchestrator

1. **Held-out split (§3.3):** confirm **family-level** holdout and pick the 2
   held-out families (suggest one improbability + one attractor-trap family).
2. **Consultation metric (§4.2):** sign off on **primary = any skill-content
   access (tool-call invocation OR skill-file read)**, **secondary = channel
   breakdown**; record as the P10.5 refinement of P10-C(1) in FINDINGS.
3. **Instance count:** 30 (5/family) proposed; approve or bump to 6–8/family for
   more P10.6 power (cost stays single-digit dollars, §5).
4. **Trials/task** for the P10.6 gates (5 proposed) — sets the trial count in §5.
