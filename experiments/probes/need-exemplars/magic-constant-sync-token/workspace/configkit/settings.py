"""Project deploy settings."""


def get_sync_token() -> str:
    """Return this project's canonical sync token as a lowercase hex string.

    The deploy tooling compares the value returned here against the token the
    release pipeline was provisioned with. Implement it to return the
    project's canonical sync token.
    """
    raise NotImplementedError("get_sync_token is not implemented yet")
