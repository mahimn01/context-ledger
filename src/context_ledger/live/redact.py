"""Redaction of keys from everything the harness logs, stores or prints.

`redact` replaces every registered Secret's value, and anything shaped like an Anthropic key
(`sk-ant-...`), with "[REDACTED-KEY]". `install_log_redaction` puts a redacting filter on the
root, anthropic, httpx and httpx2 loggers (and the root's handlers) and forces the library
loggers to WARNING. API errors are stored only as {status, error_type, request_id, message},
the message redacted; never headers, never the request object.
"""

from __future__ import annotations

import logging
import re

from .keys import Secret

KEY_SHAPE = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")
REDACTED = "[REDACTED-KEY]"
LIBRARY_LOGGERS = ("anthropic", "httpx", "httpx2")
_registered: list[Secret] = []


def register_secret(secret: Secret) -> None:
    """Redact this secret's value everywhere from now on (in this process)."""
    if not any(s is secret for s in _registered):
        _registered.append(secret)


def forget_secrets() -> None:
    """Drop every registered secret (end of a live run, and tests)."""
    _registered.clear()


def redact(text: str, secrets: tuple[Secret, ...] | list[Secret] = ()) -> str:
    for secret in (*_registered, *secrets):
        text = secret.scrub(text, REDACTED)
    return KEY_SHAPE.sub(REDACTED, text)


def safe_message(exc: BaseException) -> str:
    """The exception as "<Type>: <message>", redacted."""
    return redact(f"{type(exc).__name__}: {exc}")


def api_error_record(exc: BaseException) -> dict:
    """{status, error_type, request_id, message} of an SDK (or fake) API error, redacted."""
    status = getattr(exc, "status_code", None)
    error_type = getattr(exc, "type", None)
    request_id = getattr(exc, "request_id", None)
    return {
        "status": status if isinstance(status, int) and not isinstance(status, bool) else None,
        "error_type": redact(error_type) if isinstance(error_type, str) else type(exc).__name__,
        "request_id": redact(request_id) if isinstance(request_id, str) else None,
        "message": redact(str(exc)),
    }


class RedactingFilter(logging.Filter):
    """Rewrites each record's message (and any exception or stack text) through `redact`."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        if record.exc_info:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redact(record.exc_text)
        if record.stack_info:
            record.stack_info = redact(record.stack_info)
        return True


_FILTER = RedactingFilter()


def install_log_redaction() -> RedactingFilter:
    """Install the redacting filter (idempotent) and quiet the HTTP and SDK loggers."""
    root = logging.getLogger()
    for logger in (root, *(logging.getLogger(name) for name in LIBRARY_LOGGERS)):
        if _FILTER not in logger.filters:
            logger.addFilter(_FILTER)
    for name in LIBRARY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for handler in root.handlers:
        if _FILTER not in handler.filters:
            handler.addFilter(_FILTER)
    return _FILTER
