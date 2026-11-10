from __future__ import annotations

import hashlib
import socket
import subprocess
from pathlib import Path

import pytest

from context_ledger.anchor.text import make_text_anchor
from context_ledger.fold import make_fold
from context_ledger.ids import new_id, span_id
from context_ledger.model import Segment, Source, Span
from context_ledger.store import Ledger, MemoryBlobStore
from context_ledger.window import WindowState, apply_event, exposures, manifest_hash, read_map


_REAL_RUN = subprocess.run
APERTURE_LEDGER = Path(__file__).resolve().parents[1] / ".clp-spend" / "spend.jsonl"


def ledger_digest(path: Path) -> str | None:
    """The sha256 of a spend ledger's bytes (read only), or None if it does not exist."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


@pytest.fixture(scope="session", autouse=True)
def aperture_ledger_untouched():
    """BUILD-KEY: every test spends in a temporary directory; the real aperture ledger
    (.clp-spend/spend.jsonl) is byte-identical after the whole suite."""
    before = ledger_digest(APERTURE_LEDGER)
    yield
    assert ledger_digest(APERTURE_LEDGER) == before, "the suite changed the real aperture spend ledger"


def _no_security_tool(args, *rest, **kwargs):
    """subprocess.run, except that any argv whose first element ends in "security" (the macOS
    environment credential tool) fails: no test can ever read the real key."""
    argv0 = args[0] if isinstance(args, (list, tuple)) and args else str(args).split(" ")[0]
    if str(argv0).endswith("security"):
        raise RuntimeError("the security tool is disabled in tests")
    return _REAL_RUN(args, *rest, **kwargs)


@pytest.fixture(autouse=True)
def no_network_no_keys(monkeypatch, tmp_path_factory):
    """Every test runs with sockets disabled, the environment credential tool blocked, and no Anthropic
    credentials or config in reach."""

    def refuse(*args, **kwargs):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(subprocess, "run", _no_security_tool)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_PROFILE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path_factory.mktemp("anthropic-config")))


@pytest.fixture
def ledger():
    with Ledger(":memory:") as led:
        yield led


@pytest.fixture
def blobs():
    return MemoryBlobStore()


class Builder:
    """Writes well-formed events straight to a ledger, for testing window and fold logic
    before (and independently of) Session."""

    def __init__(self) -> None:
        self.ledger = Ledger(":memory:")
        self.blobs = MemoryBlobStore()
        self.state = WindowState()

    def emit(self, type_: str, payload: dict):
        event = self.ledger.append(type_, payload)
        apply_event(self.state, event)
        return event

    def source(self, uri: str, text: str) -> Source:
        src = Source(new_id("src", uri), uri, "v1", self.blobs.put(text.encode("utf-8")), "text/plain")
        self.emit("source_registered", {"source": src.to_dict()})
        return src

    def span(self, source: Source, start: int, end: int) -> Span:
        content_hash = self.state.sources[source.source_id].content_hash
        anchor = make_text_anchor(self.blobs.get(content_hash).decode("utf-8"), start, end)
        span = Span(
            span_id(content_hash, anchor.to_dict()),
            source.source_id,
            content_hash,
            anchor,
            self.blobs.put(anchor.exact.encode("utf-8")),
            len(anchor.exact),
        )
        self.emit("span_registered", {"span": span.to_dict()})
        return span

    def append(self, span: Span, role: str = "tool", pinned: bool = False, parent: str | None = None) -> Segment:
        seg = Segment(new_id("seg", self.ledger.head()), role, "span", span.span_id, "verbatim", pinned, span.text_hash, parent)
        self.emit("segment_appended", {"segment": seg.to_dict()})
        return seg

    def raw(self, role: str, text: str, pinned: bool = False) -> Segment:
        h = self.blobs.put(text.encode("utf-8"))
        seg = Segment(new_id("seg", self.ledger.head()), role, "raw", h, "verbatim", pinned, h, None)
        self.emit("segment_appended", {"segment": seg.to_dict()})
        return seg

    def call(self) -> str:
        call_id = new_id("call", self.ledger.head())
        self.emit(
            "call",
            {
                "call_id": call_id,
                "manifest_hash": manifest_hash(self.state),
                "exposures": exposures(self.state),
                "read_map": read_map(self.state),
            },
        )
        return call_id

    def fold(self, victims, sentences, k=None, trigger="model", policy=None):
        if k is None:
            k = min(i for i, s in enumerate(self.state.segments) if s.segment_id in set(victims))
        record = make_fold(self.state, k, victims, sentences, trigger, policy, blobs=self.blobs)
        self.emit("fold", {"fold": record.to_dict()})
        return record

    def evict(self, segment_ids) -> None:
        self.emit("evict", {"segment_ids": list(segment_ids), "reason": "test"})

    def update(self, source: Source, text: str) -> None:
        self.emit("source_updated", {"source_id": source.source_id, "content_hash": self.blobs.put(text.encode("utf-8"))})

    def events(self):
        return self.ledger.events()


@pytest.fixture
def builder():
    b = Builder()
    yield b
    b.ledger.close()
