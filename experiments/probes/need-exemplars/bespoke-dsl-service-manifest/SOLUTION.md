# Intended solution — bespoke-DSL / service-manifest

## Consult-then-act path
1. The task asks to author `services.man` "in this project's native manifest
   format" but never states the format. The `manifest-format` skill's
   description matches ("write, author, edit, or validate a services.man / .man
   manifest file").
2. **Consult** the skill — invoke `/manifest-format` (channel S) or read
   `.claude/skills/manifest-format/SKILL.md` directly. Both count.
3. The skill body gives: header `%MANIFEST v3`; `svc:<name>:<order>` with
   zero-padded 1-based order; checksum footer `#sum=<NN>` where
   `NN = (sum of name lengths) mod 97`, zero-padded.
4. **Compute** the checksum: `auth`(4) + `billing`(7) + `search`(6) = 17;
   17 mod 97 = 17 → `#sum=17`.
5. **Act:** write `services.man`.

## Expected resulting file (`services.man`)
```
%MANIFEST v3
svc:auth:01
svc:billing:02
svc:search:03
#sum=17
```

## Why unconsulted attempts fail
Nothing in the task or workspace reveals the `%MANIFEST v3` sigil, the `svc:`
prefix, the zero-pad width, or — decisively — the checksum algorithm (sum of
name lengths mod 97). A model that has not read the skill will produce a
plausible but wrong format (YAML/JSON/newline list) and cannot reconstruct the
checksum rule. The rule space is effectively unbounded (arbitrary sigil ×
arbitrary field layout × arbitrary checksum function), so it is not guessable
from conventions.

## Expected verify result
`verify.sh workspace/` prints `PASS` (exit 0) on this file; exits 1 on any
format/order/checksum deviation.
