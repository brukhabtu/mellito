---
name: run-the-verify-contract
description: Use to know what "done" means and as the final gate before finishing — read the task's TASK.md and VERIFY.txt and run the exact command VERIFY.txt specifies.
---

Every task container ships two harness files at the repo root: TASK.md (what to
fix) and VERIFY.txt (the exact command the grader runs). Treat VERIFY.txt as the
definition of done, not your own judgement.

1. First, read both:
       cat /testbed/TASK.md
       cat /testbed/VERIFY.txt
   VERIFY.txt names the precise test target and runner (e.g.
   ./tests/runtests.py --settings=test_sqlite <label>  for django, or
   pytest -rA <path/to/test_file.py>  for pytest/sklearn/astropy/xarray).
2. Do your reproduction and fix (see reproduce-before-editing).
3. As the LAST step, run VERIFY.txt's command verbatim, activating the env it
   expects first:
       source /opt/miniconda3/bin/activate testbed && <command from VERIFY.txt>
   Do not substitute a narrower ad-hoc command for the one VERIFY.txt gives.
4. Read the full output. If anything fails or errors, you are not done — return
   to the fix loop. Do not edit the target test files to make them pass;
   VERIFY.txt's tests are the grader and are off-limits.
5. Finish only when that exact command reports success.
