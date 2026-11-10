"""The live factory: the only place a real SDK client is built and the only `reveal()` call.

The key is passed explicitly, so the SDK never looks for one in the environment or on disk,
and no environment variable is ever set from it. The base URL is passed explicitly too, so an
environment override cannot send the key anywhere else. Nothing in the build or the tests
executes this function.
"""

from __future__ import annotations

from .keys import Secret

API_BASE_URL = "https://api.anthropic.com"


def make_live_client(secret: Secret):
    """A real `anthropic.Anthropic` client: no SDK retries (the harness retries through the
    SpendGuard), a 120 s timeout."""
    import anthropic

    return anthropic.Anthropic(api_key=secret.reveal(), base_url=API_BASE_URL, max_retries=0, timeout=120)
