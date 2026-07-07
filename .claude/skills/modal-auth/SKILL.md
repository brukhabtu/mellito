---
description: >-
  Authenticate the Modal CLI in this environment (needed before any serving,
  smoke, or sweep runs on Modal). Use when `modal profile current` is `default`
  / unset, when `modal run`/`modal deploy` hangs or errors on auth, or when
  setting up a fresh Claude Code container. Handles the proxied, browserless
  case: installs the proxy shim, prints the auth URL for the operator, waits for
  authorisation, stores + verifies the token.
argument-hint: ""
arguments: []
---

# Authenticate Modal (headless, behind the agent proxy)

Goal: end with `modal profile current` naming a real workspace and
`~/.modal.toml` present, so `infra/modal_app.py` (serve / smoke / sweeps) can run.

Modal creds do **not** arrive via GitHub repo secrets (those reach only Actions
runners) — this session's env comes from the Claude Code environment config.
This skill uses an interactive browser login instead, which needs nothing
pre-provisioned.

## Steps

1. **Short-circuit if already authed.** Run `modal profile current`. If it prints
   a workspace name (not `default`) and `~/.modal.toml` exists, report "already
   authenticated as <workspace>" and stop.

2. **Ensure prerequisites.** `modal` must be importable and, critically,
   `python-socks` must be installed — Modal only tunnels its gRPC through
   `HTTPS_PROXY` when python-socks is present; without it the client hangs
   silently. Install both if missing:
   ```bash
   pip3 install -q modal "python-socks[asyncio]"
   ```
   (Confirm `HTTPS_PROXY` is set in the env; on Claude Code web it always is.)

3. **Run the login driver in the background** (it polls for minutes, so don't
   block on it), capturing output to a log:
   ```bash
   nohup env PYTHONUNBUFFERED=1 python3 .claude/skills/modal-auth/login.py \
     > /tmp/modal-auth.log 2>&1 &
   ```
   Wait ~15s, then read `/tmp/modal-auth.log` and extract the `AUTH_URL=` line.

4. **Give the operator the URL.** Surface the `AUTH_URL` value and ask them to
   open it and authorise. The browser callback targets localhost inside this
   container and won't reach them — that's fine; the driver completes via
   polling regardless. Do **not** ask for any code or token to be pasted back.

5. **Wait for completion.** Poll `/tmp/modal-auth.log` for `SUCCESS` (token
   stored + verified) or `TIMEOUT`. On success, confirm with
   `modal profile current`. On timeout, the flow can be re-run.

## Notes
- The token lives in `~/.modal.toml` (never committed; it's per-container and
  does not persist across fresh sessions — re-run this skill after a restart).
- If `modal run`/`deploy` later fails with a connection hang, the usual cause is
  a fresh container missing `python-socks` — reinstall (step 2) and retry.
