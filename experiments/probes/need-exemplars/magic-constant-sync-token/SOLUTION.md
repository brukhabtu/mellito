# Intended solution — magic-constant / sync-token

## Consult-then-act path
1. The task asks for "the project's canonical sync token" — a fixed constant it
   explicitly says not to invent. The `deploy-constants` skill's description
   matches ("emit, return, validate, or look up the project's sync token").
2. **Consult** the skill — either invoke it (`/deploy-constants`, channel S) or
   read `.claude/skills/deploy-constants/SKILL.md` directly. Both count as
   consultation (see design doc §Anti-gaming).
3. The skill body gives the literal: canonical sync token = `9f3ac1d70b2e`.
4. **Act:** edit `configkit/settings.py` so `get_sync_token()` returns
   `"9f3ac1d70b2e"`.

## Expected resulting code
```python
def get_sync_token() -> str:
    return "9f3ac1d70b2e"
```

## Why unconsulted attempts fail
The token is an arbitrary 12-hex literal (~2^48 ≈ 2.8×10^14 space). No amount of
reasoning over the task or workspace recovers it; a guess passes with
probability ~1/2.8×10^14. `verify.sh` compares a salted SHA-256 digest, so the
value is absent from the verifier and cannot be lifted from it.

## Expected verify result
`verify.sh workspace/` prints `PASS` and exits 0 on this solution; it exits 1 on
the unimplemented stub or on any wrong token.
