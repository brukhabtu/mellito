#!/usr/bin/env bash
# HARNESS-SIDE verifier for the magic-constant family (exemplar: sync-token).
# NOT materialized into the worker workspace (same contract as hidden_tests):
# the worker never sees this file, so the expected value cannot be read off it.
#
# Anti-leak: the token itself does NOT appear here. We compare a salted SHA-256
# digest of the returned value. Reading this file reveals only the digest, which
# cannot be inverted, and with a 12-hex (~2^48) constant a brute-force preimage
# search is infeasible even if the digest leaked.
#
# Usage: verify.sh [WORKSPACE_DIR]   (defaults to ./workspace)
# Exit 0 = pass, nonzero = fail.
set -u
DIR="${1:-$(cd "$(dirname "$0")" && pwd)/workspace}"
SALT="mellito-p10-need"
EXPECT="0ec96104f840067bd2e97fc6fb784f1ef5d35cd8642d4d62d0e9c0ef59c9b7bf"

python3 - "$DIR" "$SALT" "$EXPECT" <<'PY'
import sys, hashlib, importlib.util, pathlib
workdir, salt, expect = sys.argv[1], sys.argv[2], sys.argv[3]
mod_path = pathlib.Path(workdir) / "configkit" / "settings.py"
if not mod_path.exists():
    print("FAIL: configkit/settings.py not found"); sys.exit(1)
spec = importlib.util.spec_from_file_location("_ck_settings", mod_path)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    val = mod.get_sync_token()
except Exception as e:
    print(f"FAIL: get_sync_token() raised {type(e).__name__}: {e}"); sys.exit(1)
if not isinstance(val, str):
    print(f"FAIL: expected str, got {type(val).__name__}"); sys.exit(1)
digest = hashlib.sha256((salt + ":" + val).encode()).hexdigest()
if digest == expect:
    print("PASS"); sys.exit(0)
print("FAIL: returned value does not match the canonical sync token"); sys.exit(1)
PY
