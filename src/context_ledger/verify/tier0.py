"""Tier 0: a claim passes only if it cites, directly, a span that is verbatim in the window."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Claim, Knowledge
from ..window import WindowState, down, kappa, ref_kind_of

ACTIONS = frozenset({"accept", "refetch", "abstain", "flag"})
_REFETCHABLE = frozenset({Knowledge.SUMMARY_ONLY, Knowledge.DROPPED, Knowledge.STALE})


@dataclass(frozen=True)
class Verdict:
    """Outcome of verification. `tier` is the last tier run. `support` holds the span ids that
    passed, or on a refetch verdict the span ids the harness must re-fetch. `detail` (v2) holds
    the offending claim tokens, in claim order, of a value_not_covered or
    identifier_not_anchored verdict; it is written to the dict only when non-empty, so older
    verdicts (and ledgers) read and hash as before."""

    tier: int
    passed: bool
    action: str
    reason: str
    support: tuple[str, ...] = ()
    detail: tuple[str, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "support", tuple(self.support))
        object.__setattr__(self, "detail", tuple(self.detail))
        if self.action not in ACTIONS:
            raise ValueError(f"unknown verdict action {self.action!r}")

    def to_dict(self) -> dict:
        d = {"tier": self.tier, "passed": self.passed, "action": self.action, "reason": self.reason, "support": list(self.support)}
        if self.detail:
            d["detail"] = list(self.detail)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Verdict:
        return cls(d["tier"], d["passed"], d["action"], d["reason"], tuple(d["support"]), detail=tuple(d.get("detail", ())))


REFETCH_REASONS = ("stale", "dropped", "summary_only", "refetch_obligation")


@dataclass(frozen=True)
class RefetchRequired(Verdict):
    """The typed verdict of a `needs_refetch` claim (returned, never raised): the claim's cited
    spans are held only as summary_only, dropped or stale, or it has unmet refetch obligations
    (reason "refetch_obligation"). `obligations` are the canonical span ids to re-fetch,
    sorted; they are also the verdict's `support`."""

    claim_id: str = ""
    obligations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "obligations", tuple(self.obligations))
        if self.action != "refetch" or self.reason not in REFETCH_REASONS:
            raise ValueError(f"RefetchRequired needs action refetch and a reason in {REFETCH_REASONS}")

    def to_dict(self) -> dict:
        return {**super().to_dict(), "obligations": list(self.obligations)}


def tier0(claim: Claim, state: WindowState) -> Verdict:
    """Pass if some span in the union of down(cite) is verbatim and is itself cited directly.

    No citations: flag "no_citations". Otherwise, if the down set is empty (only inference or
    raw cites): flag. Else refetch, with the summary-only, dropped or stale spans as support."""
    if not claim.citations:
        return Verdict(0, False, "flag", "no_citations")
    refs = [c.ref for c in claim.citations]
    reach = set().union(*(down(state, ref) for ref in refs))
    direct = {ref for ref in refs if ref_kind_of(ref) == "span"}
    states = {span_id: kappa(state, span_id) for span_id in reach}
    passing = sorted(s for s in reach & direct if states[s] is Knowledge.VERBATIM)
    if passing:
        return Verdict(0, True, "accept", "verbatim_cited", tuple(passing))
    if not reach:
        return Verdict(0, False, "flag", "no_span_support")
    return Verdict(0, False, "refetch", "needs_refetch", tuple(sorted(s for s in reach if states[s] in _REFETCHABLE)))
