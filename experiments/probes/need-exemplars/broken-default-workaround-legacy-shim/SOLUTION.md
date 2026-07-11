# Intended solution — broken-default-workaround / legacy-shim

## Consult-then-act path
1. `run()` crashes with "fastpath unavailable in this build". The task says to
   use the project's *supported workaround* and warns that merely returning the
   right value is not enough. The `legacy-shim` skill's description matches
   (disabled fastpath, "fastpath unavailable" error, CI-safe build).
2. **Consult** the skill — invoke `/legacy-shim` (channel S) or read
   `.claude/skills/legacy-shim/SKILL.md` directly. Both count.
3. The skill gives the two-part workaround: (1) import `transform` from
   `dataproc.compat`; (2) create `.procrc` in the root containing exactly
   `use_legacy_transform = true`.
4. **Act:** edit `pipeline.py`'s import and create `.procrc`.

## Expected resulting state
`pipeline.py`:
```python
from dataproc.compat import transform


def run():
    return transform([1, 2, 3])
```
`.procrc`:
```
use_legacy_transform = true
```

## Why unconsulted attempts fail
The default path is *visibly* broken, and a competent model can discover
`dataproc.compat` by reading the workspace and make `run()` return `[2, 4, 6]`
the natural way — but that is exactly the "naive fix" the task rules out. The
load-bearing fact is the `.procrc` marker gate: its filename and exact content
(`use_legacy_transform = true`) are a project-specific convention present
nowhere in the task or workspace, and the verifier ("CI gate") is harness-side,
so there is no feedback loop that would reveal the missing marker. The model
believes it is done while `verify.sh` fails on the marker check. Guessing the
marker is an arbitrary-identifier search, not a convention.

## Expected verify result
`verify.sh workspace/` prints `PASS` (exit 0) only when BOTH the marker is
present/correct AND `run()` returns `[2, 4, 6]`; a compat-only fix without
`.procrc` exits 1 at the marker check.
