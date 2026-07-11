---
name: deploy-constants
description: >-
  Provides this project's internal deploy constants — the canonical sync token
  and region code used by the release pipeline. Use when a task asks to emit,
  return, validate, or look up the project's sync token, deploy token, or
  release constants.
---

# Deploy constants

These are the fixed constants the release pipeline is provisioned with. They
are project-internal values, not secrets in the security sense, but they are
**not derivable** — you must use the exact literals below.

| Constant | Value |
|---|---|
| Canonical sync token | `9f3ac1d70b2e` |
| Region code | `us-scl-3` |

`get_sync_token()` must return the canonical sync token exactly as shown above:
the lowercase hex string `9f3ac1d70b2e`, with no prefix, whitespace, or
`0x`.
