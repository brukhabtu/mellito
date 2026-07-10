#!/usr/bin/env python3
"""Stop hook: refuse to end the worker's turn while it has produced no source
edit. The v002 completion contract is prose the model disobeys (it ends the
turn asking a nonexistent user for direction — see FINDINGS 2026-07-08). This
converts the "never stop without a diff" clause from an instruction into a
harness-enforced gate.

Gate signal: a non-empty `git status --porcelain` over /testbed, EXCLUDING the
harness files (.claude/, VERIFY.txt, TASK.md). We deliberately do NOT gate on
VERIFY.txt exiting 0 — the hidden test patch is not present in the worker
sandbox, so verify-green is not observable here; "a real source edit exists" is
the reliable, in-sandbox proxy and it targets the exact failure mode (every
observed loss ended with an empty diff).

Bounded: at most MAX_NUDGES blocks per episode (counter kept under the excluded
.claude/ dir), after which the stop is allowed through — so a genuinely stuck
worker fails (empty_diff) rather than burning turns into a timeout. Reads the
Claude Code Stop-hook JSON on stdin; emits a block decision as JSON on stdout.
"""
import json
import subprocess
import sys

MAX_NUDGES = 4
COUNTER = "/testbed/.claude/.stop_nudges"
TESTBED = "/testbed"

try:
    json.load(sys.stdin)  # drain the hook payload; we don't need its fields
except Exception:
    pass

try:
    with open(COUNTER) as f:
        n = int(f.read().strip() or "0")
except Exception:
    n = 0

# Any tracked-or-untracked change outside the harness scaffolding counts as
# "the worker acted". Pathspec exclude magic keeps .claude/VERIFY.txt/TASK.md
# (and this counter file, which lives under .claude/) out of the signal.
try:
    r = subprocess.run(
        ["git", "-C", TESTBED, "status", "--porcelain", "--",
         ":(exclude).claude", ":(exclude)VERIFY.txt", ":(exclude)TASK.md"],
        capture_output=True, text=True, timeout=30)
    has_edit = bool(r.stdout.strip())
except Exception:
    # If git can't be queried, don't trap the worker — let it stop.
    has_edit = True

if has_edit or n >= MAX_NUDGES:
    sys.exit(0)  # allow the turn to end

try:
    with open(COUNTER, "w") as f:
        f.write(str(n + 1))
except Exception:
    pass

print(json.dumps({
    "decision": "block",
    "reason": (
        "You are ending your turn but no source file has been edited yet, so "
        "the task is NOT done. You are running fully autonomously: there is no "
        "human to answer questions, approve steps, or clarify anything — any "
        "message asking for direction is a dead end that no one will read. "
        "Apply your fix now with the Edit or Write tool (an explanation is not "
        "an edit; only a changed file on disk counts), then run the command in "
        "VERIFY.txt. Keep going until the working tree contains a real source "
        "edit."
    ),
}))
sys.exit(0)
