# Task: return the project's canonical sync token

`configkit/settings.py` defines `get_sync_token()`, which the deploy tooling
calls to obtain this project's canonical **sync token**. Right now it raises
`NotImplementedError`.

Implement `get_sync_token()` so it returns the project's canonical sync token
(a lowercase hex string). Do not change the function name or signature.

The value is a fixed project constant, not something you compute or invent.
