"""The session-note table (BUILD-V4 item 1c): (service, attribute) -> the notes that state it, in
the order the conversation showed them; the latest one wins (an update supersedes the note).

Built only from the text of the notes the user showed (never a question's gold or metadata),
with the gate's own parse: a note states a pair for each "the <attribute> of service <service>"
it holds, and a later note supersedes an earlier one exactly when it states the same exact
service token and attribute phrase (verify.gate.superseding_note's rule). Lookups are exact;
pairs sharing only the service or only the attribute are listed as DIFFERENT pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..anchor.text import normalize_for_match
from ..verify.gate import _NOTE_PAIR


@dataclass
class NoteTable:
    chains: dict[tuple[str, str], list[str]] = field(default_factory=dict)  # (service, attribute) -> note refs, oldest first
    order: list[str] = field(default_factory=list)  # every note ref, as shown

    def add(self, ref: str, text: str) -> None:
        """Record note `ref` (e.g. "note:5") with its user-visible text."""
        self.order.append(ref)
        for m in _NOTE_PAIR.finditer(normalize_for_match(text)):
            chain = self.chains.setdefault((m["service"], m["attribute"]), [])
            if not chain or chain[-1] != ref:
                chain.append(ref)

    def chain(self, service: str, attribute: str) -> list[str]:
        """The notes stating exactly this pair, oldest first (the last is the current one)."""
        return list(self.chains.get((service, attribute), []))

    def latest(self, service: str, attribute: str) -> str | None:
        chain = self.chains.get((service, attribute))
        return chain[-1] if chain else None

    def different(self, service: str, attribute: str, limit: int = 4) -> list[tuple[str, str, str]]:
        """(service, attribute, latest note) of noted pairs sharing the service or the attribute
        but not both: look-alikes, never the pair asked."""
        out = [(s, a, refs[-1]) for (s, a), refs in self.chains.items() if (s, a) != (service, attribute) and (s == service or a == attribute)]
        return sorted(out, key=lambda x: (x[0] != service, x))[:limit]


__all__ = ["NoteTable"]
