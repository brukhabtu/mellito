---
name: reproduce-before-editing
description: Use before editing any source file — write and run a minimal reproduction of the reported bug first, then re-run that same script after the fix to confirm it.
---

Reproduce the failure with a throwaway script BEFORE touching source, and keep
that script as your fast pass/fail signal.

1. Read TASK.md (or the issue text) and lift the smallest snippet that triggers
   the bug into a runnable script. Prefer an inline command over a repo file:
       python -c "
       <imports from the issue>
       <the minimal call the issue says is broken>
       print('REPRO_OK')"
   For multi-line cases use a heredoc to /tmp (never inside a tests/ dir, never
   named test_*.py):  cat > /tmp/repro.py << 'EOF' ... EOF ; python /tmp/repro.py
2. Run it and CONFIRM it fails the way the issue describes (read the actual
   error/traceback text, not just the exit code). If it does not fail, you have
   the wrong reproduction — fix the script before editing any source.
3. Make the source edit.
4. Re-run the IDENTICAL script. Add a print after each step so a silent wrong
   result can't masquerade as success ('First fit OK' / 'REPRO_OK').
5. If it still fails, do not pile on more edits — re-localize (see the
   localize-by-symbol procedure) and repeat from step 3.
6. Only once the repro prints its success marker, proceed to the full verify.
