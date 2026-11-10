"""Records of the Context Ledger Protocol.

Every record is a frozen dataclass whose `to_dict()` output is plain canonical-JSON data
(dicts, lists, str, int, bool, None) and whose `from_dict()` inverts it exactly, so
`X.from_dict(json.loads(canonical_json(x.to_dict()))) == x`. `from_dict` validates types and
raises ValueError, TypeError or KeyError on malformed input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Union


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Knowledge(str, Enum):
    """Knowledge state kappa_t(sigma) of a span at a call, strongest first; STALE overrides all."""

    VERBATIM = "verbatim"
    SNIPPET = "snippet"
    SUMMARY_ONLY = "summary_only"
    DROPPED = "dropped"
    UNSEEN = "unseen"
    STALE = "stale"

    @property
    def rank(self) -> int:
        """verbatim 5, snippet 4, summary_only 3, dropped 2, unseen 1, stale 0."""
        return _KNOWLEDGE_RANK[self.value]


_KNOWLEDGE_RANK = {"verbatim": 5, "snippet": 4, "summary_only": 3, "dropped": 2, "unseen": 1, "stale": 0}

ROLES = frozenset(r.value for r in Role)
REF_KINDS = frozenset({"span", "fold", "raw"})
FIDELITIES = frozenset({"verbatim", "summary"})
NODE_KINDS = frozenset({"function", "class", "assign"})
TRIGGERS = frozenset({"model", "backstop"})
CONTENT_TYPES = frozenset({"text", "blocks"})
CLAIM_STATUSES = frozenset({"grounded", "ungrounded", "inference", "abstain", "needs_refetch"})


def _str(d: dict, key: str) -> str:
    value = d[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a str, got {type(value).__name__}")
    return value


def _opt_str(d: dict, key: str) -> str | None:
    value = d[key]
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{key} must be a str or None, got {type(value).__name__}")
    return value


def _int(d: dict, key: str) -> int:
    value = d[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an int, got {type(value).__name__}")
    return value


def _bool(d: dict, key: str) -> bool:
    value = d[key]
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a bool, got {type(value).__name__}")
    return value


def _list(d: dict, key: str) -> list:
    value = d[key]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{key} must be a list, got {type(value).__name__}")
    return list(value)


def _dict(obj: Any, what: str) -> dict:
    if not isinstance(obj, dict):
        raise TypeError(f"{what} must be a dict, got {type(obj).__name__}")
    return obj


def _one_of(value: str, allowed: frozenset[str], what: str) -> None:
    if value not in allowed:
        raise ValueError(f"{what} must be one of {sorted(allowed)}, got {value!r}")


@dataclass(frozen=True)
class Source:
    source_id: str
    uri: str
    version: str
    content_hash: str
    media_type: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Source:
        d = _dict(d, "source")
        return cls(_str(d, "source_id"), _str(d, "uri"), _str(d, "version"), _str(d, "content_hash"), _str(d, "media_type"))


@dataclass(frozen=True, kw_only=True)
class TextAnchor:
    """Character range [start, end) of newline-normalized source text, with context."""

    kind: str = "text"
    start: int
    end: int
    exact: str
    prefix: str
    suffix: str

    def __post_init__(self) -> None:
        if self.kind != "text":
            raise ValueError(f"TextAnchor kind must be 'text', got {self.kind!r}")
        if not 0 <= self.start <= self.end:
            raise ValueError(f"bad TextAnchor range {self.start}..{self.end}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TextAnchor:
        d = _dict(d, "anchor")
        return cls(
            kind=_str(d, "kind"),
            start=_int(d, "start"),
            end=_int(d, "end"),
            exact=_str(d, "exact"),
            prefix=_str(d, "prefix"),
            suffix=_str(d, "suffix"),
        )


@dataclass(frozen=True, kw_only=True)
class PyAnchor:
    """A Python symbol (function, class or module-level assignment) and its character range."""

    kind: str = "python"
    path: str
    qualname: str
    node_kind: str
    ast_hash: str
    start: int
    end: int
    commit: str | None = None

    def __post_init__(self) -> None:
        if self.kind != "python":
            raise ValueError(f"PyAnchor kind must be 'python', got {self.kind!r}")
        _one_of(self.node_kind, NODE_KINDS, "node_kind")
        if not 0 <= self.start <= self.end:
            raise ValueError(f"bad PyAnchor range {self.start}..{self.end}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PyAnchor:
        d = _dict(d, "anchor")
        return cls(
            kind=_str(d, "kind"),
            path=_str(d, "path"),
            qualname=_str(d, "qualname"),
            node_kind=_str(d, "node_kind"),
            ast_hash=_str(d, "ast_hash"),
            start=_int(d, "start"),
            end=_int(d, "end"),
            commit=_opt_str(d, "commit"),
        )


Anchor = Union[TextAnchor, PyAnchor]


def anchor_from_dict(d: dict) -> Anchor:
    """Rebuild a TextAnchor or PyAnchor from its dict, dispatching on `kind`."""
    kind = _dict(d, "anchor").get("kind")
    if kind == "text":
        return TextAnchor.from_dict(d)
    if kind == "python":
        return PyAnchor.from_dict(d)
    raise ValueError(f"unknown anchor kind {kind!r}")


@dataclass(frozen=True)
class Span:
    span_id: str
    source_id: str
    source_hash: str
    anchor: Anchor
    text_hash: str
    n_chars: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Span:
        d = _dict(d, "span")
        return cls(
            _str(d, "span_id"),
            _str(d, "source_id"),
            _str(d, "source_hash"),
            anchor_from_dict(d["anchor"]),
            _str(d, "text_hash"),
            _int(d, "n_chars"),
        )


@dataclass(frozen=True)
class Segment:
    """One entry of the window. `parent` set on a verbatim span segment makes it a snippet.

    `content_type` "blocks" means the blob is the canonical JSON of a list of Anthropic content
    blocks (M2 uses it only for assistant messages with text and/or tool_use blocks).
    `tool_use_id` set means the segment is rendered inside the tool_result answering that tool
    use; `is_error` marks that tool_result as an error (only with `tool_use_id`). The three
    fields are omitted from the dict at their defaults, so M1 segments are unchanged."""

    segment_id: str
    role: str
    ref_kind: str
    ref: str
    fidelity: str
    pinned: bool
    text_hash: str
    parent: str | None = None
    content_type: str = "text"
    tool_use_id: str | None = None
    is_error: bool = False

    def __post_init__(self) -> None:
        _one_of(self.role, ROLES, "role")
        _one_of(self.ref_kind, REF_KINDS, "ref_kind")
        _one_of(self.fidelity, FIDELITIES, "fidelity")
        _one_of(self.content_type, CONTENT_TYPES, "content_type")
        if self.content_type == "blocks" and (self.role != "assistant" or self.ref_kind != "raw"):
            raise ValueError("only raw assistant segments hold content blocks")
        if self.is_error and self.tool_use_id is None:
            raise ValueError("is_error needs a tool_use_id")

    def to_dict(self) -> dict:
        d = asdict(self)
        for key, default in (("content_type", "text"), ("tool_use_id", None), ("is_error", False)):
            if d[key] == default:
                del d[key]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Segment:
        d = _dict(d, "segment")
        return cls(
            _str(d, "segment_id"),
            _str(d, "role"),
            _str(d, "ref_kind"),
            _str(d, "ref"),
            _str(d, "fidelity"),
            _bool(d, "pinned"),
            _str(d, "text_hash"),
            _opt_str(d, "parent"),
            _str(d, "content_type") if "content_type" in d else "text",
            _opt_str(d, "tool_use_id") if "tool_use_id" in d else None,
            _bool(d, "is_error") if "is_error" in d else False,
        )


@dataclass(frozen=True)
class SummarySentence:
    """One sentence of a fold summary. Each cite is a span id, "fold:<fold_id>#<i>",
    "raw:<segment_id>" or "inference"."""

    text: str
    cites: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cites", tuple(self.cites))

    def to_dict(self) -> dict:
        return {"text": self.text, "cites": list(self.cites)}

    @classmethod
    def from_dict(cls, d: dict) -> SummarySentence:
        d = _dict(d, "sentence")
        cites = _list(d, "cites")
        if not all(isinstance(c, str) for c in cites):
            raise TypeError("cites must be strings")
        return cls(_str(d, "text"), tuple(cites))


@dataclass(frozen=True)
class FoldRecord:
    """A fold: `victims` left the window and one summary segment was inserted at index `k`.
    `policy` is None for model folds, or {"id": "backstop-v0"|"backstop-v1", ...} for backstop
    folds. `aliases` lists the [alias, target] pairs the fold introduces, in counter order (its
    whole-fold handle, its sentences, then any cited ref without an alias yet); it is empty for
    M1 folds, whose summary text cites full refs instead of aliases."""

    fold_id: str
    k: int
    victims: tuple[str, ...]
    sentences: tuple[SummarySentence, ...]
    trigger: str
    policy: dict | None
    freed_tokens: int
    summary_segment_id: str
    summary_hash: str
    aliases: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "victims", tuple(self.victims))
        object.__setattr__(self, "sentences", tuple(self.sentences))
        object.__setattr__(self, "aliases", tuple(tuple(pair) for pair in self.aliases))
        _one_of(self.trigger, TRIGGERS, "trigger")

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "k": self.k,
            "victims": list(self.victims),
            "sentences": [s.to_dict() for s in self.sentences],
            "trigger": self.trigger,
            "policy": None if self.policy is None else dict(self.policy),
            "freed_tokens": self.freed_tokens,
            "summary_segment_id": self.summary_segment_id,
            "summary_hash": self.summary_hash,
            **({"aliases": [list(pair) for pair in self.aliases]} if self.aliases else {}),
        }

    @classmethod
    def from_dict(cls, d: dict) -> FoldRecord:
        d = _dict(d, "fold")
        victims = _list(d, "victims")
        if not all(isinstance(v, str) for v in victims):
            raise TypeError("victims must be strings")
        policy = d["policy"]
        if policy is not None:
            policy = dict(_dict(policy, "policy"))
        return cls(
            _str(d, "fold_id"),
            _int(d, "k"),
            tuple(victims),
            tuple(SummarySentence.from_dict(s) for s in _list(d, "sentences")),
            _str(d, "trigger"),
            policy,
            _int(d, "freed_tokens"),
            _str(d, "summary_segment_id"),
            _str(d, "summary_hash"),
            _alias_pairs(d.get("aliases", [])),
        )


def _alias_pairs(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or not all(
        isinstance(p, list) and len(p) == 2 and all(isinstance(x, str) for x in p) for p in value
    ):
        raise TypeError("aliases must be a list of [alias, target] string pairs")
    return tuple((a, t) for a, t in value)


@dataclass(frozen=True)
class Citation:
    """`ref` is always canonical; `alias` records the alias the model wrote, if it wrote one."""

    ref: str
    quote: str | None = None
    alias: str | None = None

    def to_dict(self) -> dict:
        d = {"ref": self.ref, "quote": self.quote}
        return {**d, "alias": self.alias} if self.alias is not None else d

    @classmethod
    def from_dict(cls, d: dict) -> Citation:
        d = _dict(d, "citation")
        return cls(_str(d, "ref"), _opt_str(d, "quote"), _opt_str(d, "alias") if "alias" in d else None)


@dataclass(frozen=True)
class Claim:
    """A model claim. `verdict` is the recorded Verdict.to_dict() or None. `turn_id` names the
    answer turn it belongs to; `supersedes` names an earlier needs_refetch or ungrounded claim
    that this one replaces. Both are omitted from the dict when None (M1 claims have neither)."""

    claim_id: str
    call_id: str | None
    text: str
    citations: tuple[Citation, ...]
    status: str
    verdict: dict | None = None
    turn_id: str | None = None
    supersedes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "citations", tuple(self.citations))
        _one_of(self.status, CLAIM_STATUSES, "status")

    def to_dict(self) -> dict:
        d = {
            "claim_id": self.claim_id,
            "call_id": self.call_id,
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "status": self.status,
            "verdict": None if self.verdict is None else dict(self.verdict),
        }
        for key in ("turn_id", "supersedes"):
            if getattr(self, key) is not None:
                d[key] = getattr(self, key)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Claim:
        d = _dict(d, "claim")
        verdict = d["verdict"]
        if verdict is not None:
            verdict = dict(_dict(verdict, "verdict"))
        return cls(
            _str(d, "claim_id"),
            _opt_str(d, "call_id"),
            _str(d, "text"),
            tuple(Citation.from_dict(c) for c in _list(d, "citations")),
            _str(d, "status"),
            verdict,
            _opt_str(d, "turn_id") if "turn_id" in d else None,
            _opt_str(d, "supersedes") if "supersedes" in d else None,
        )


@dataclass(frozen=True)
class Event:
    """One ledger entry: hash = sha256(bytes.fromhex(prev_hash) + canonical_json({seq, type, payload}))."""

    seq: int
    type: str
    payload: dict
    prev_hash: str
    hash: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Event:
        d = _dict(d, "event")
        return cls(_int(d, "seq"), _str(d, "type"), _dict(d["payload"], "payload"), _str(d, "prev_hash"), _str(d, "hash"))


EVENT_TYPES = frozenset(
    {
        "source_registered",
        "source_updated",
        "span_registered",
        "segment_appended",
        "fold",
        "evict",
        "call",
        "refetch",
        "claim",
        "alias_assigned",
        "pricing_set",
        "answer_emitted",
        "model_call",
        "tokenizer_set",
    }
)
