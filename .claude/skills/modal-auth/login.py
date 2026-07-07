#!/usr/bin/env python3
"""Headless Modal web-auth that works behind the agent proxy.

Standard `modal token new` assumes a local browser and a localhost callback,
neither of which exists in a remote/proxied Claude Code container. This driver:
  - starts Modal's token flow and prints the auth URL as plain text (so it can
    be surfaced to the operator — Modal's own CLI writes it to the TTY via rich,
    which a pipe/redirect can't capture);
  - polls TokenFlowWait (the cross-machine path) until the operator authorises
    on modal.com — the localhost callback is skipped, not needed;
  - stores + verifies the token into ~/.modal.toml.

Prereqs the skill installs first: `modal` and `python-socks[asyncio]` (Modal
routes its gRPC through HTTP_PROXY/HTTPS_PROXY only when python-socks is present;
without it the connection retries forever and hangs silently).

Prints one line `AUTH_URL=<url>` for the caller to extract, and finally
`SUCCESS` (exit 0) or `TIMEOUT` (exit 2).
"""
import asyncio
import sys

from modal.client import _Client
from modal.config import config
from modal.token_flow import _TokenFlow, _set_token

POLL_ATTEMPTS = 30  # * ~40s ≈ 20 min window


async def main() -> int:
    server_url = config.get("server_url")
    print(f"[modal-auth] connecting to {server_url} via proxy ...", flush=True)
    async with _Client.anonymous(server_url) as client:
        tf = _TokenFlow(client)
        async with tf.start("cli") as (_flow_id, web_url, code):
            print("READY", flush=True)
            print(f"AUTH_URL={web_url}", flush=True)
            print(f"CODE={code or '(none)'}", flush=True)
            print("[modal-auth] waiting for browser authorisation ...", flush=True)
            result = None
            for attempt in range(POLL_ATTEMPTS):
                result = await tf.finish(timeout=40.0)
                if result is not None:
                    break
                print(f"[modal-auth] still waiting (attempt {attempt + 2}) ...",
                      flush=True)
            if result is None:
                print("TIMEOUT: no authorisation within the window", flush=True)
                return 2
        srv = client.server_url
        ws = getattr(result, "workspace_username", "") or "?"
        print(f"[modal-auth] authorised; workspace={ws}", flush=True)
    await _set_token(result.token_id, result.token_secret,
                     activate=True, verify=True, server_url=srv)
    print("SUCCESS: token stored + verified in ~/.modal.toml", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
