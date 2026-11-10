"""Public-export credential adapter.

The private platform credential integration is intentionally not distributed.
Only explicit environment values supplied by the person running a new experiment
are accepted. No command is executed and no credential file is read.
The experiment, scoring and budget-guard logic are unchanged.
"""
from __future__ import annotations
import os
from typing import Protocol
from collections.abc import Mapping

DEFAULT_PROFILE = "research"
PROFILES = ("research", "aperture")
MASK = "Secret(****)"

class KeyUnavailable(Exception):
    def __init__(self, code: int | str) -> None:
        self.code = code
        super().__init__("the explicit experiment credential is unavailable")

class Secret:
    """A key held in memory only. Every printable form is "Secret(****)"; it cannot be pickled
    or copied, and it has no __eq__ (so no timing-sensitive comparison). `reveal()` returns the
    value and is called in exactly one place, the live factory. `scrub` and `occurs_in` let
    redaction and leak scans use the value without revealing it."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError("a Secret needs a non-empty string")
        self._value = value

    def __repr__(self) -> str:
        return MASK

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return MASK

    def __reduce__(self):
        raise TypeError("a Secret cannot be pickled or copied")

    def __reduce_ex__(self, protocol):
        raise TypeError("a Secret cannot be pickled or copied")

    def reveal(self) -> str:
        return self._value

    def scrub(self, text: str, replacement: str) -> str:
        """`text` with every occurrence of the value replaced."""
        return text.replace(self._value, replacement)

    def occurs_in(self, data: bytes) -> bool:
        """Whether the value's UTF-8 bytes occur in `data` (for leak scans)."""
        return self._value.encode("utf-8") in data

class KeyProvider(Protocol):
    profile: str  # the key profile, which also picks the spend ledger and caps

    def get(self) -> Secret: ...

class FakeKeyProvider:
    """A provider for tests: returns a Secret of the value it was given, under `profile`."""

    def __init__(self, value: str, profile: str = DEFAULT_PROFILE) -> None:
        self._secret = Secret(value)
        self.profile = profile
        self.calls = 0

    def get(self) -> Secret:
        self.calls += 1
        return self._secret

class EnvironmentKeyProvider:
    """Read only a profile-specific variable, with no cross-profile fallback."""
    def __init__(self, profile: str = DEFAULT_PROFILE, *, environ: Mapping[str, str] | None = None) -> None:
        if profile not in PROFILES:
            raise ValueError("unknown experiment credential profile")
        self.profile = profile
        self._environ = environ

    def get(self) -> Secret:
        values = os.environ if self._environ is None else self._environ
        name = "CONTEXT_LEDGER_RESEARCH_API_KEY" if self.profile == "research" else "CONTEXT_LEDGER_LEGACY_API_KEY"
        value = values.get(name)
        if not value:
            raise KeyUnavailable("missing explicit environment value")
        return Secret(value)
