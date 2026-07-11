#!/usr/bin/env bash
# HARNESS-SIDE verifier for the bespoke-DSL family (exemplar: service-manifest).
# NOT materialized into the worker workspace (same contract as hidden_tests).
#
# NOTE: unlike the magic-constant family, this verifier must embed the grammar
# it checks, so it cannot be salted/hashed — its non-leakability rests ENTIRELY
# on the hidden-verify materialization contract (worker never sees this file).
# See design doc §Anti-gaming, "DSL family caveat".
#
# Usage: verify.sh [WORKSPACE_DIR]   (defaults to ./workspace)
# Exit 0 = pass, nonzero = fail.
set -u
DIR="${1:-$(cd "$(dirname "$0")" && pwd)/workspace}"

python3 - "$DIR" <<'PY'
import sys, pathlib
workdir = pathlib.Path(sys.argv[1])
required = ["auth", "billing", "search"]   # required services, in order

p = workdir / "services.man"
if not p.exists():
    print("FAIL: services.man not found"); sys.exit(1)
lines = p.read_text().split("\n")
# tolerate a single trailing empty line from a final newline
if lines and lines[-1] == "":
    lines = lines[:-1]

if not lines or lines[0] != "%MANIFEST v3":
    print("FAIL: header line must be exactly '%MANIFEST v3'"); sys.exit(1)
if len(lines) < 2 + 1:  # header + >=1 svc + checksum
    print("FAIL: manifest too short"); sys.exit(1)

svc_lines = lines[1:-1]
footer = lines[-1]

if len(svc_lines) != len(required):
    print(f"FAIL: expected {len(required)} service lines, got {len(svc_lines)}"); sys.exit(1)

for idx, (line, name) in enumerate(zip(svc_lines, required), start=1):
    expect = f"svc:{name}:{idx:02d}"
    if line != expect:
        print(f"FAIL: service line {idx} is {line!r}, expected {expect!r}"); sys.exit(1)

checksum = sum(len(n) for n in required) % 97
expect_footer = f"#sum={checksum:02d}"
if footer != expect_footer:
    print(f"FAIL: checksum footer is {footer!r}, expected {expect_footer!r}"); sys.exit(1)

print("PASS"); sys.exit(0)
PY
