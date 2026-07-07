---
description: >-
  Run a piece of work as a bounded goal: pin acceptance criteria before
  starting, validate against them when done. Use for any non-trivial unit of
  implementation or experiment work — "implement the smoke suite", "build the
  trial runner", "run the next search cycle" — or whenever asked to "take this
  on as a goal".
argument-hint: "[work description]"
---

# Run work as a bounded goal

Work: $ARGUMENTS

1. **Pin criteria first.** Before any implementation, write 2–5 acceptance
   criteria — concrete, checkable, stated so failure is possible. Record them
   in the commit message body (`goal:` prefix) or the goal description if
   using /goal.
2. **Do the work.** Explore, plan, act as needed. If the work turns out to
   need different criteria, stop and re-pin explicitly — never silently
   redefine done.
3. **Validate honestly.** Check the deliverable against each pinned
   criterion; run every mechanical check (tests, smoke, status.py) rather
   than asserting it. Partial passes are reported as partial.
4. **Record.** One entry in findings/FINDINGS.md: goal, criteria,
   pass/fail per criterion, run IDs where applicable.

Scope guard: one goal moves one thing forward. If the criteria list is
describing two deliverables, split into two goals.
