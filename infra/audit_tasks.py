#!/usr/bin/env python3
"""Mechanical corpus audit: re-checks every admitted task spec in tasks/dev
and tasks/staging against the schema and admission rules in tasks/schema.md
and the corpus-curator checklist.

This is a cheap, offline, no-Docker sanity net — it re-verifies the *paper
trail* (fields, split legality, digest pinning, hidden-tests presence, oracle
leakage) that a human or import script could have gotten wrong, not the
hermetic 6/6 determinism verdict itself (that's determinism_check.py, and it
needs Docker). Run this before trusting a corpus for a sweep.

tasks/holdout is never read here: it is sealed (hook-enforced) and this script
has no business touching it — auditing dev/staging is sufficient to catch
spec-authoring mistakes before they propagate.

Usage:
  python3 infra/audit_tasks.py

Exit 0 iff zero failures across dev + staging.
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["dev", "staging"]  # holdout is sealed; never audited here

REQUIRED_FIELDS = [
    "id", "provenance", "source", "image", "description",
    "verify", "timeout_s", "hidden_tests", "admitted",
]
VALID_PROVENANCE = {
    "public-pretrained", "own-repo", "post-cutoff", "held-out-public",
}
SPLIT_PROVENANCE = {
    "dev": {"public-pretrained", "own-repo"},
    "staging": {"held-out-public", "post-cutoff", "own-repo"},
}
DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}$")
TIMEOUT_MIN, TIMEOUT_MAX = 60, 7200


def _fail(failures, task_id, reason):
    failures.append((task_id, reason))


def audit_task(task_dir: Path, split: str, failures: list) -> None:
    task_id = task_dir.name
    yaml_path = task_dir / "task.yaml"
    try:
        spec = yaml.safe_load(yaml_path.read_text()) or {}
    except Exception as e:
        _fail(failures, task_id, f"task.yaml failed to parse: {e}")
        return
    if not isinstance(spec, dict):
        _fail(failures, task_id, "task.yaml did not parse to a mapping")
        return

    # 1. required fields present and non-empty
    for field in REQUIRED_FIELDS:
        if field not in spec or spec[field] in (None, "", [], {}):
            _fail(failures, task_id, f"missing or empty required field: {field}")
    admitted = spec.get("admitted") or {}
    if isinstance(admitted, dict):
        if admitted.get("determinism_check") != "6/6":
            _fail(failures, task_id,
                  f"admitted.determinism_check != '6/6' (got {admitted.get('determinism_check')!r})")
    else:
        _fail(failures, task_id, "admitted is not a mapping")

    # 2. id matches directory name
    if spec.get("id") != task_id:
        _fail(failures, task_id, f"id {spec.get('id')!r} != directory name {task_id!r}")

    # 3. provenance legality (global + per-split)
    provenance = spec.get("provenance")
    if provenance is not None and provenance not in VALID_PROVENANCE:
        _fail(failures, task_id, f"invalid provenance: {provenance!r}")
    elif provenance is not None and provenance not in SPLIT_PROVENANCE[split]:
        _fail(failures, task_id,
              f"provenance {provenance!r} not allowed in {split} "
              f"(allowed: {sorted(SPLIT_PROVENANCE[split])})")

    # 4. image is digest-pinned
    image = spec.get("image")
    if isinstance(image, str) and not DIGEST_RE.search(image):
        _fail(failures, task_id, f"image not digest-pinned: {image!r}")

    # 5. hidden_tests file exists, non-empty, looks like a unified diff
    hidden_tests = spec.get("hidden_tests")
    tests_text = None
    if isinstance(hidden_tests, str) and hidden_tests:
        tests_path = task_dir / hidden_tests
        if not tests_path.exists():
            _fail(failures, task_id, f"hidden_tests file missing: {hidden_tests}")
        else:
            tests_text = tests_path.read_text()
            if not tests_text.strip():
                _fail(failures, task_id, f"hidden_tests file is empty: {hidden_tests}")
            elif not any(
                line.startswith("diff --git ") or line.startswith("--- ")
                for line in tests_text.splitlines()
            ):
                _fail(failures, task_id,
                      f"hidden_tests file doesn't look like a unified diff: {hidden_tests}")

    # 6. timeout_s is an int within [60, 7200]
    timeout_s = spec.get("timeout_s")
    if not isinstance(timeout_s, int) or isinstance(timeout_s, bool) or not (
        TIMEOUT_MIN <= timeout_s <= TIMEOUT_MAX
    ):
        _fail(failures, task_id, f"timeout_s out of range or not an int: {timeout_s!r}")

    # 7. oracle-leak check. A hard FAIL only for verdict-revealing content:
    # hidden-test *names* (def test_xxx) appearing in the description. Generic
    # added-line overlap is a WARN, not a failure — SWE-bench tests are built
    # from the issue's own repro snippet, so imports/repro calls legitimately
    # appear in both (adjudicated benign on this corpus, 2026-07-07 findings).
    description = spec.get("description") or ""
    if tests_text and isinstance(description, str):
        test_names = re.findall(r"^\+\s*def (test_\w+)", tests_text, re.M)
        leaked = [n for n in test_names if n in description]
        if leaked:
            _fail(failures, task_id,
                  f"oracle leak: hidden-test name(s) in description: {leaked[:3]}")
        else:
            for line in tests_text.splitlines():
                if not line.startswith("+") or line.startswith("+++"):
                    continue
                added = line[1:].strip()
                if len(added) < 20 or not added.strip("{} \t"):
                    continue
                if added in description:
                    print(f"WARN {task_id}: repro-snippet overlap with hidden "
                          f"tests (benign unless it names a test): {added[:60]!r}")
                    break

    # 8. verify is a non-empty string
    verify = spec.get("verify")
    if not isinstance(verify, str) or not verify.strip():
        _fail(failures, task_id, f"verify is not a non-empty string: {verify!r}")


def main() -> int:
    failures = []
    counts = {}
    for split in SPLITS:
        split_dir = ROOT / "tasks" / split
        task_dirs = sorted(p.parent for p in split_dir.glob("*/task.yaml"))
        counts[split] = len(task_dirs)
        for task_dir in task_dirs:
            audit_task(task_dir, split, failures)

    for task_id, reason in failures:
        print(f"FAIL {task_id}: {reason}")

    total = sum(counts.values())
    print(
        f"audited {total} specs (dev={counts.get('dev', 0)} "
        f"staging={counts.get('staging', 0)}): {len(failures)} failures"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
