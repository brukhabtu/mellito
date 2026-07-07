#!/usr/bin/env python3
"""Corpus admission: the 3+3 determinism check (corpus-curator item 2).

A task is admissible only if its verify command gives an *identical, correct*
verdict across blind reruns: 3x in the broken state (must fail every time) and
3x in the known-good solution state (must pass every time) — 6/6 consistent.
Flaky verdicts (a broken state that sometimes passes, or a solution that
sometimes fails) mean the task is not hermetic; reject it, never admit.

This runs the check inside the task's pinned container image via Docker, so the
verdict path is binary and offline (no LLM). It is source-agnostic: SWE-bench /
R2E-Gym / SWE-Gym imports and own-repo bugs all express the broken and solution
states as shell command lists in a small JSON descriptor.

Descriptor (JSON on stdin or a file arg):
{
  "id": "getmoto__moto-7365",
  "image": "sha256:...|repo@sha256:...|repo:tag",   # pinned digest preferred
  "workdir": "/testbed",                              # where setup runs
  "setup":   ["git checkout <base_commit>", ...],     # reach base state (opt)
  "to_broken":   ["git apply /tmp/test.patch"],        # base -> broken (tests only)
  "to_solution": ["git apply /tmp/gold.patch"],        # broken -> solution (+ fix)
  "files": {"/tmp/test.patch": "<contents>", ...},     # written into container
  "verify": "python -m pytest -x -q tests/foo.py::bar",# exit 0 = pass
  "timeout_s": 1800
}

Usage:
  python3 infra/determinism_check.py path/to/descriptor.json
  cat descriptor.json | python3 infra/determinism_check.py -

Exit 0 iff 6/6 consistent (admissible). Nonzero otherwise. The JSON verdict is
printed to stdout for the caller / findings log.
"""
import json
import shlex
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

RERUNS = 3  # per state; corpus-curator mandates 3+3


def _run(cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class Container:
    """A long-lived container we exec into, so setup work is done once and each
    verify rerun sees the same filesystem state (a fresh `docker run` per rerun
    would redo setup and hide nondeterminism)."""

    def __init__(self, image, workdir):
        self.name = f"det-{uuid.uuid4().hex[:12]}"
        self.workdir = workdir
        # Keep it alive; no network (hermetic verdict path).
        r = _run([
            "docker", "run", "-d", "--name", self.name, "--network", "none",
            "-w", workdir, "--entrypoint", "sleep", image, "infinity",
        ])
        if r.returncode != 0:
            raise RuntimeError(f"docker run failed: {r.stderr.strip()}")

    def put(self, path, contents):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(contents)
            host = f.name
        # Ensure parent dir exists in the container, then copy.
        parent = str(Path(path).parent)
        _run(["docker", "exec", self.name, "mkdir", "-p", parent])
        r = _run(["docker", "cp", host, f"{self.name}:{path}"])
        Path(host).unlink(missing_ok=True)
        if r.returncode != 0:
            raise RuntimeError(f"docker cp failed: {r.stderr.strip()}")

    def sh(self, script, timeout=None, check=True):
        r = _run(
            ["docker", "exec", "-w", self.workdir, self.name, "bash", "-lc", script],
            timeout=timeout,
        )
        if check and r.returncode != 0:
            raise RuntimeError(
                f"setup step failed ({script!r}): rc={r.returncode}\n{r.stderr[-800:]}"
            )
        return r

    def verify(self, cmd, timeout):
        """Return the binary verdict of one verify run: True=pass (exit 0)."""
        try:
            r = self.sh(cmd, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return False  # a hang is not a pass
        return r.returncode == 0

    def snapshot(self, tag):
        _run(["docker", "commit", self.name, tag])

    def remove(self):
        _run(["docker", "rm", "-f", self.name])


def check(desc: dict) -> dict:
    image = desc["image"]
    workdir = desc.get("workdir", "/")
    verify = desc["verify"]
    timeout = int(desc.get("timeout_s", 1800))

    # Broken container: setup -> to_broken. Then snapshot so the solution state
    # starts from an identical base and only adds the fix.
    broken = Container(image, workdir)
    result = {"id": desc.get("id"), "image": image, "reruns": RERUNS}
    try:
        for path, contents in (desc.get("files") or {}).items():
            broken.put(path, contents)
        for step in desc.get("setup") or []:
            broken.sh(step, timeout=timeout)
        base_tag = f"det-base-{uuid.uuid4().hex[:10]}"
        broken.snapshot(base_tag)
        for step in desc.get("to_broken") or []:
            broken.sh(step, timeout=timeout)

        broken_verdicts = [broken.verify(verify, timeout) for _ in range(RERUNS)]

        # Solution container from the identical base snapshot: to_broken +
        # to_solution (the gold fix). Same tests, plus the fix.
        sol = Container(base_tag, workdir)
        try:
            for step in (desc.get("to_broken") or []) + (desc.get("to_solution") or []):
                sol.sh(step, timeout=timeout)
            sol_verdicts = [sol.verify(verify, timeout) for _ in range(RERUNS)]
        finally:
            sol.remove()
            _run(["docker", "rmi", "-f", base_tag])
    finally:
        broken.remove()

    broken_fail = all(v is False for v in broken_verdicts)  # broken must fail 3/3
    sol_pass = all(v is True for v in sol_verdicts)          # solution must pass 3/3
    admissible = broken_fail and sol_pass
    consistent = len(set(broken_verdicts)) == 1 and len(set(sol_verdicts)) == 1

    result.update({
        "broken_verdicts": ["pass" if v else "fail" for v in broken_verdicts],
        "solution_verdicts": ["pass" if v else "fail" for v in sol_verdicts],
        "broken_fails_3of3": broken_fail,
        "solution_passes_3of3": sol_pass,
        "consistent_6of6": consistent and admissible,
        "admissible": admissible,
        "determinism_check": f"{sum(v is False for v in broken_verdicts) + sum(sol_verdicts)}/6",
    })
    return result


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else "-"
    raw = sys.stdin.read() if arg == "-" else Path(arg).read_text()
    desc = json.loads(raw)
    result = check(desc)
    print(json.dumps(result, indent=2))
    return 0 if result["admissible"] else 1


if __name__ == "__main__":
    sys.exit(main())
