"""Fast transform path. Disabled by design in CI builds (upstream bug DP-4471)."""


def transform(data):
    raise RuntimeError("fastpath unavailable in this build")
