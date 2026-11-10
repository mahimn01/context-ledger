"""Text anchors: character ranges with exact text and context, re-resolvable after edits.

All offsets are character offsets into newline-normalized text. Callers normalize first;
these functions never normalize behind the caller's back, so offsets stay meaningful.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..model import Anchor, TextAnchor

RESOLUTION_STATUSES = frozenset({"OK", "MOVED", "STALE", "DANGLING"})
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Resolution:
    """Result of re-resolving an anchor against new text.

    OK: same place. MOVED: same content elsewhere (`anchor` is the new anchor). STALE: the
    content changed or is ambiguous. DANGLING: the target no longer exists."""

    status: str
    anchor: Anchor
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in RESOLUTION_STATUSES:
            raise ValueError(f"bad resolution status {self.status!r}")


def normalize_newlines(text: str) -> str:
    """Return `text` with "\\r\\n" and "\\r" replaced by "\\n"."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_for_match(text: str) -> str:
    """NFKC-normalize, collapse every whitespace run to one space, and strip.

    Used only for Tier 1 quote matching, never for offsets."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def make_text_anchor(text: str, start: int, end: int, context: int = 32) -> TextAnchor:
    """Anchor the non-empty range [start, end) of `text` with up to `context` chars each side.

    Raises ValueError unless 0 <= start < end <= len(text)."""
    if not 0 <= start < end <= len(text):
        raise ValueError(f"bad range {start}..{end} for text of length {len(text)}")
    return TextAnchor(
        start=start,
        end=end,
        exact=text[start:end],
        prefix=text[max(0, start - context) : start],
        suffix=text[end : end + context],
    )


def _occurrences(text: str, needle: str) -> list[int]:
    """Start offsets of every (possibly overlapping) occurrence of a non-empty `needle`."""
    found, i = [], text.find(needle)
    while i != -1:
        found.append(i)
        i = text.find(needle, i + 1)
    return found


def resolve_text(anchor: TextAnchor, text: str) -> Resolution:
    """Re-resolve `anchor` against `text`.

    OK if text[start:end] == exact. Otherwise one occurrence of `exact` gives MOVED (offsets,
    prefix and suffix recomputed); several give MOVED only if exactly one also matches both
    prefix and suffix, else STALE "ambiguous"; none gives STALE."""
    if text[anchor.start : anchor.end] == anchor.exact:
        return Resolution("OK", anchor)
    hits = _occurrences(text, anchor.exact) if anchor.exact else []
    if len(hits) > 1:
        hits = [
            i
            for i in hits
            if text[max(0, i - len(anchor.prefix)) : i] == anchor.prefix
            and text[i + len(anchor.exact) : i + len(anchor.exact) + len(anchor.suffix)] == anchor.suffix
        ]
        if len(hits) != 1:
            return Resolution("STALE", anchor, "ambiguous")
    if not hits:
        return Resolution("STALE", anchor, "not_found")
    start = hits[0]
    return Resolution("MOVED", make_text_anchor(text, start, start + len(anchor.exact)))


def chunk_views(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Split `text` into views (global start, end) of at most `size` chars, consecutive views
    sharing `overlap` chars, covering the whole text. Chunks are views, never copies, so an
    anchor built at chunk start + relative offsets equals the anchor built at global offsets.

    Raises ValueError unless size > 0 and 0 <= overlap < size."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError(f"need size > 0 and 0 <= overlap < size, got size={size} overlap={overlap}")
    views: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        views.append((start, end))
        if end == len(text):
            break
        start = end - overlap
    return views
