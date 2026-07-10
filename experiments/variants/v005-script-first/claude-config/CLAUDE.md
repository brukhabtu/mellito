# Worker instructions

You are resolving one self-contained coding task in this repository.

- The task description states the goal; `VERIFY.txt` in the workspace root
  contains the exact command that decides success. Run it to check your work
  as often as useful; you are done only when it exits 0.
- Plan your own approach: explore, reproduce, fix, verify. Retry with a
  different approach if verification fails.
- Do not modify test files or anything referenced by the verify command
  unless the task explicitly says to.

## Work through scripts

Drive this task the way you would drive your own pipeline: write scripts and
run them with the Bash tool. That loop — script, run, read output, adjust — is
how progress happens here.

- First write a small reproduction script (e.g. `repro.py`) that triggers the
  bug, run it, and read the real traceback before touching any source file.
- Apply source changes however is most natural from Bash — a heredoc, a
  `python` one-liner rewriting the file, a patch file applied with `git apply`
  or `patch`. The Edit and Write tools also work; use whatever puts the change
  on disk fastest.
- Then re-run your reproduction script. Keep cycling script → output → change
  until the reproduction behaves, and finish with the VERIFY.txt command.
- Scratch scripts are yours: keep them in the workspace root, no cleanup
  needed. Don't name them `test_*.py` (the test runner would collect them).
