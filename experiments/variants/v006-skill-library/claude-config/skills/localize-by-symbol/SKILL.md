---
name: localize-by-symbol
description: Use when you know the failing function/class/attribute name (from the traceback or issue) but not which file to edit — grep the symbol, read the defining frame, and edit there.
---

Jump to the code that actually raises by chasing the symbol, not by guessing
files.

1. From the reproduction's traceback, take the DEEPEST frame you own (the last
   "File .../pkg/....py", line N, in <name>" before the exception) — that name
   is your symbol. If there is no traceback yet, take the API name from the
   issue.
2. Locate its definition (not just call sites):
       grep -rn "def <symbol>\|class <symbol>" <package_dir>/
   Widen to plain  grep -rn "<symbol>"  only if the def/class grep is empty.
   Use Glob for a filename you can guess:  **/<module>*.py
3. Read the defining file around that line. Confirm the code path the traceback
   walked actually reaches this function before editing.
4. Make the smallest edit at that site. If a helper it calls is the real cause,
   repeat steps 2-3 on that helper's name rather than patching the caller.
5. Re-run the reproduction (reproduce-before-editing). If the same traceback
   points at a DIFFERENT frame now, that new frame's name is your next symbol —
   go back to step 2. If the fix made things worse, revert the edit
   (git checkout -- <file>) and re-localize instead of stacking changes.
