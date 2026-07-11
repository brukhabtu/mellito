#!/usr/bin/env bash
# HARNESS-SIDE verifier for the broken-default-workaround family (exemplar:
# legacy-shim). NOT materialized into the worker workspace (hidden_tests
# contract) — this is the "CI gate" the task refers to. The worker sees only the
# broken default (fastpath raises) and the skill; it never sees this gate logic,
# so the required marker cannot be read off it.
#
# The marker's expected content is checked via a salted SHA-256 digest, so the
# literal never appears in this file.
#
# Usage: verify.sh [WORKSPACE_DIR]   (defaults to ./workspace)
# Exit 0 = pass, nonzero = fail.
set -u
DIR="${1:-$(cd "$(dirname "$0")" && pwd)/workspace}"
SALT="mellito-p10-need"
EXPECT_MARKER="8b64e1a24e204e8b3f8fb92093cec1f2bf38fe802c216a2deb58113e1679a216"

python3 - "$DIR" "$SALT" "$EXPECT_MARKER" <<'PY'
import sys, hashlib, importlib, pathlib
workdir = pathlib.Path(sys.argv[1]).resolve()
salt, expect_marker = sys.argv[2], sys.argv[3]

# Part 2: the legacy marker file (checked FIRST — it is the load-bearing,
# skill-only fact; a naive compat-only fix fails here).
procrc = workdir / ".procrc"
if not procrc.exists():
    print("FAIL: CI gate rejects change — .procrc marker missing"); sys.exit(1)
line = procrc.read_text().strip()
if hashlib.sha256((salt + ":" + line).encode()).hexdigest() != expect_marker:
    print("FAIL: CI gate rejects change — .procrc marker content wrong"); sys.exit(1)

# Part 1 + correctness: run() must return [2,4,6] via the compat path.
sys.path.insert(0, str(workdir))
for m in list(sys.modules):
    if m == "pipeline" or m.startswith("dataproc"):
        del sys.modules[m]
try:
    pipeline = importlib.import_module("pipeline")
    result = pipeline.run()
except Exception as e:
    print(f"FAIL: pipeline.run() raised {type(e).__name__}: {e}"); sys.exit(1)
if result != [2, 4, 6]:
    print(f"FAIL: run() returned {result!r}, expected [2, 4, 6]"); sys.exit(1)

print("PASS"); sys.exit(0)
PY
