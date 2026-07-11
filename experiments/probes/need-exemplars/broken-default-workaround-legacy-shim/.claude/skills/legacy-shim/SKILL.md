---
name: legacy-shim
description: >-
  Documents this project's supported workaround for builds where the fast
  transform path (dataproc.fastpath) is disabled. Use when a task involves a
  disabled fastpath, a crashing pipeline.run(), the "fastpath unavailable"
  error, or making dataproc run in a CI-safe build.
---

# Disabled-fastpath workaround

`dataproc.fastpath` is intentionally disabled in CI-safe builds (upstream bug
DP-4471). The **supported** workaround has TWO required parts. Doing only one of
them leaves the project's CI gate rejecting the change.

1. **Switch the import.** In `pipeline.py`, import `transform` from
   `dataproc.compat` instead of `dataproc.fastpath`. `compat.transform` produces
   identical output.

2. **Enable the legacy marker.** The compat path is gated behind a project
   marker file. Create a file named `.procrc` in the workspace root containing
   **exactly** this single line:

   ```
   use_legacy_transform = true
   ```

Both parts are mandatory. Without the `.procrc` marker the CI gate treats the
compat import as an unsanctioned change and rejects it, even though `run()`
would return the right value locally.
