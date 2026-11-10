"""Claim verification: Tier 0 (knowledge state) then Tier 1 (quotes and values)."""

from __future__ import annotations

from ..model import Claim
from ..store.blobs import BlobStore
from ..window import WindowState
from .tier0 import REFETCH_REASONS, RefetchRequired, Verdict, tier0
from .tier1 import (
    Witness,
    identifiers_witnessed,
    is_anchored,
    missing_identifiers,
    missing_values,
    module_prefixed,
    numeric_value,
    strip_source_paths,
    tier1,
    value_tokens,
    value_tokens_typed,
    values_covered,
)


def verify_claim(claim: Claim, state: WindowState, blobs: BlobStore) -> Verdict:
    """Run Tier 0, then Tier 1 only if Tier 0 passed; return the last verdict."""
    verdict = tier0(claim, state)
    return tier1(claim, state, blobs) if verdict.passed else verdict


__all__ = [
    "REFETCH_REASONS",
    "RefetchRequired",
    "Verdict",
    "Witness",
    "identifiers_witnessed",
    "is_anchored",
    "missing_identifiers",
    "missing_values",
    "module_prefixed",
    "numeric_value",
    "strip_source_paths",
    "tier0",
    "tier1",
    "value_tokens",
    "value_tokens_typed",
    "values_covered",
    "verify_claim",
]
