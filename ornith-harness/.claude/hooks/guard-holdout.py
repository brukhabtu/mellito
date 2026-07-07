#!/usr/bin/env python3
"""PreToolUse guard: block any read/access of tasks/holdout/**.

The optimization loop must never see holdout tasks. This is reward-hacking
defense layer 2 (deterministic monitor), applied to ourselves. Exit 2 blocks
the tool call; stderr is shown to Claude as the reason.

Escape hatch: the final holdout evaluation (G4 gate) is run by the human
operator, who creates .holdout-unlocked at repo root for the duration of
that run and deletes it after. The hook logs every unlock use.
"""
import json
import os
import sys
from datetime import datetime, timezone

HOLDOUT_MARKER = "tasks/holdout"


def main() -> int:
    event = json.load(sys.stdin)
    tool = event.get("tool_name", "")
    ti = event.get("tool_input", {}) or {}

    # Collect every string the tool call could use to reach a path.
    candidates = []
    for key in ("file_path", "path", "pattern", "command", "notebook_path"):
        v = ti.get(key)
        if isinstance(v, str):
            candidates.append(v)
    if isinstance(ti.get("edits"), list):
        candidates.extend(str(e) for e in ti["edits"])

    touched = any(HOLDOUT_MARKER in c for c in candidates)
    if not touched:
        return 0

    root = os.environ.get("CLAUDE_PROJECT_DIR", ".")
    unlock = os.path.join(root, ".holdout-unlocked")
    if os.path.exists(unlock):
        with open(os.path.join(root, "findings", "holdout-access.log"), "a") as f:
            f.write(
                f"{datetime.now(timezone.utc).isoformat()} UNLOCKED access via {tool}: "
                f"{candidates}\n"
            )
        return 0

    print(
        "BLOCKED: tasks/holdout/** is sealed. The holdout set must never be "
        "read, listed, grepped, or executed against during optimization. If "
        "this is the final gate evaluation, the human operator must create "
        ".holdout-unlocked first. Record this attempt as a data point; do not "
        "work around it.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
