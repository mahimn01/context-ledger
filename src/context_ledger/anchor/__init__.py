"""Anchors: stable references to text ranges and Python symbols."""

from __future__ import annotations

from .python import python_anchors, resolve_python
from .text import Resolution, chunk_views, make_text_anchor, normalize_for_match, normalize_newlines, resolve_text

__all__ = [
    "Resolution",
    "chunk_views",
    "make_text_anchor",
    "normalize_for_match",
    "normalize_newlines",
    "python_anchors",
    "resolve_python",
    "resolve_text",
]
