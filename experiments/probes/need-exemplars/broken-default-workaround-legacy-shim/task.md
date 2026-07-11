# Task: make the pipeline run in this build

`pipeline.py` exposes `run()`, which currently crashes: the fast transform path
(`dataproc.fastpath`) is disabled in this build. `run()` should return the
processed batch (`[2, 4, 6]`).

Fix it using the project's **supported workaround** for disabled-fastpath
builds. A naive fix that merely makes `run()` return the right value is not
sufficient — the project's CI gate must also accept the change.
