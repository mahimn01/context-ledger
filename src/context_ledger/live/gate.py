"""The live gate: live mode needs the --live flag and a safe environment.

Refused environments: CLAUDECODE or CLAUDE_CODE_ENTRYPOINT (running inside Claude Code),
ANTHROPIC_LOG (the SDK would log request headers), and, as extra defence, ANTHROPIC_BASE_URL
and ANTHROPIC_CUSTOM_HEADERS (the SDK would send the key elsewhere or add headers). Only the
presence of these names is checked; their values are never read.
"""

from __future__ import annotations

from collections.abc import Mapping

REFUSED_ENV = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "ANTHROPIC_LOG", "ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS")


class LiveRefused(Exception):
    """Live mode is not allowed here; `reason` says why."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"live mode refused: {reason}")
        self.reason = reason


def require_live(args: object, environ: Mapping[str, str]) -> None:
    """Raise LiveRefused unless `args.live` is True and none of REFUSED_ENV is set."""
    if getattr(args, "live", False) is not True:
        raise LiveRefused("the --live flag is not set")
    present = [name for name in REFUSED_ENV if name in environ]
    if present:
        raise LiveRefused(f"{', '.join(present)} set in the environment")
