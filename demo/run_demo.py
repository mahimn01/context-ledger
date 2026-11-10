"""End-to-end demo of the Context Ledger Protocol with a scripted mock model (no network).

What it does, in plain words:
  1. Opens a fresh ledger in a temporary folder with a small 600-token context budget.
  2. Feeds an embedded Python module and a text note into the window one span at a time,
     calling the mock model after each, so the backstop has to fold (summarize) old spans.
  3. Asks three questions. Q1's span is still verbatim, so the answer is accepted. Q2's span
     was folded, so the model guesses from the summary; the verifier demands a re-fetch, the
     span comes back verbatim, and the corrected answer is accepted. Q3 asks about a symbol
     that does not exist, so the model abstains.
  4. Edits the module (a constant changes, a function is only reformatted) and shows the
     constant's span becomes stale while the reformatted function still resolves.
  5. Checks every ledger invariant (expects zero violations), then tampers with copies of the
     ledger to show the checker catches it (P7 for a removed pinned segment, P11 for an edit).

How to run:  .venv/bin/python demo/run_demo.py   (exit code 0 means every expectation held)
What it changes: nothing outside a temporary folder that is deleted at the end (unless you
call main(db_path=...) to keep the ledger file for the CLI).
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]

from context_ledger.fold import BackstopPolicy  # noqa: E402
from context_ledger.invariants import check  # noqa: E402
from context_ledger.model import Event, PyAnchor, Span  # noqa: E402
from context_ledger.session import Session  # noqa: E402
from context_ledger.store import Ledger, SqliteBlobStore, rechain  # noqa: E402
from context_ledger.window import manifest_hash, replay, window_tokens  # noqa: E402
from mock_model import MockModel, Question  # noqa: E402

BUDGET = 600
SYSTEM_PROMPT = "Answer only from cited spans. Cite span ids."
MODULE_PATH = "inventory/client.py"
MODULE = '''\
"""HTTP client settings and retry helpers for the inventory service."""

import functools
import time

DEFAULT_TIMEOUT = 37
MAX_RETRIES = 4
USER_AGENT = "inventory-client/2.1"


def retrying(fn):
    """Retry `fn` up to MAX_RETRIES times on ConnectionError."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except ConnectionError:
                time.sleep(backoff_delay(attempt))
        return fn(*args, **kwargs)

    return wrapper


def backoff_delay(attempt):
    """Seconds to wait before retry number `attempt`."""
    return min(2 ** attempt, 300)


class InventoryClient:
    """Thin client for the inventory HTTP API."""

    def __init__(self, base_url, timeout=DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def url_for(self, sku):
        return f"{self.base_url}/items/{sku}"

    @retrying
    def fetch(self, sku):
        return {"sku": sku, "url": self.url_for(sku), "timeout": self.timeout}

    def restock_threshold(self, sku):
        """Reorder when stock falls below this many units."""
        return 12 if sku.startswith("A") else 25


@retrying
def ping(client):
    """Health check used by the deploy script."""
    return client.fetch("health")
'''
NOTE_URI = "docs/runbook.txt"
NOTE = """Deploy runbook for the inventory service.
Staging runs 3 replicas behind one load balancer; production runs 9.
Rotate the API signing key every 30 days and record it in the vault.
"""
# The order spans enter the window: the earliest ones get folded first.
APPEND_ORDER = [
    "backoff_delay",
    "retrying",
    "retrying.<locals>.wrapper",
    "InventoryClient",
    "InventoryClient.__init__",
    "InventoryClient.url_for",
    "InventoryClient.fetch",
    "InventoryClient.restock_threshold",
    "note:replicas",
    "note:rotation",
    "ping",
    "USER_AGENT",
    "MAX_RETRIES",
    "DEFAULT_TIMEOUT",
]


def label(session: Session, span: Span) -> str:
    """Human-readable span name: <uri>#<qualname> or <uri>#<start>-<end>."""
    uri = session.state.sources[span.source_id].uri
    a = span.anchor
    return f"{uri}#{a.qualname}" if isinstance(a, PyAnchor) else f"{uri}#{a.start}-{a.end}"


def edited_module() -> str:
    """DEFAULT_TIMEOUT changes value; backoff_delay is only reformatted (same AST)."""
    return MODULE.replace("DEFAULT_TIMEOUT = 37", "DEFAULT_TIMEOUT = 41").replace(
        "    return min(2 ** attempt, 300)", "    return min(\n        2 ** attempt,\n        300,\n    )"
    )


def tamper_remove_pinned(events: list[Event]) -> list[Event]:
    """Copy of the ledger in which the pinned system segment is evicted just before the first
    call and that call's manifest hash is rewritten to match, then the chain is re-hashed."""
    events = list(events)
    system_id = next(e.payload["segment"]["segment_id"] for e in events if e.type == "segment_appended")
    i = next(i for i, e in enumerate(events) if e.type == "call")
    events.insert(i, Event(0, "evict", {"segment_ids": [system_id], "reason": "tampered"}, "", ""))
    rewritten = {**events[i + 1].payload, "manifest_hash": manifest_hash(replay(events[: i + 1]))}
    events[i + 1] = replace(events[i + 1], payload=rewritten)
    return rechain(events)


def tamper_edit_payload(events: list[Event]) -> list[Event]:
    """Copy of the ledger with one claim's text edited and the chain NOT re-hashed."""
    events = list(events)
    i = next(i for i, e in enumerate(events) if e.type == "claim")
    claim = {**events[i].payload["claim"], "text": "The value is 9000."}
    events[i] = replace(events[i], payload={"claim": claim})
    return events


def ask(session: Session, model: MockModel, result, q: Question):
    """Get the mock model's answer and record it: an abstention, or a verified claim.
    Returns (answer, verdict); verdict is None when the model abstained."""
    answer = model.answer(result, q)
    if answer.kind == "abstain":
        session.abstain(answer.text)
        return answer, None
    return answer, session.claim(answer.text, answer.citations)[1]


def main(db_path: str | Path | None = None) -> dict:
    """Run the demo, print a trace and return {ok, violations, tampered_props, refetches,
    verdicts, stale_spans}."""
    if db_path is not None:
        return _run(Path(db_path))
    with tempfile.TemporaryDirectory() as tmp:
        return _run(Path(tmp) / "demo.sqlite")


def _run(db: Path) -> dict:
    trace: list[str] = []
    with Ledger(db) as ledger, SqliteBlobStore(db) as blobs:
        session = Session(ledger, blobs, BUDGET, BackstopPolicy(budget_tokens=BUDGET))
        model = MockModel(lambda sid: blobs.get(session.state.spans[sid].text_hash).decode() if sid in session.state.spans else None)

        # Pinned system prompt, an embedded module and a note.
        session.append_raw("system", SYSTEM_PROMPT, pinned=True)
        module = session.register_source(MODULE_PATH, MODULE, media_type="text/x-python")
        py_spans = {s.anchor.qualname: s for s in session.register_python_spans(module, MODULE_PATH)}
        note = session.register_source(NOTE_URI, NOTE)
        line2 = NOTE.index("Staging")
        line3 = NOTE.index("Rotate")
        spans = {
            **py_spans,
            "note:replicas": session.register_text_span(note, line2, NOTE.index("\n", line2)),
            "note:rotation": session.register_text_span(note, line3, NOTE.index("\n", line3)),
        }
        trace.append(f"registered {len(py_spans)} Python spans and 2 text spans; budget {BUDGET} tokens")

        # Append spans one by one, calling after each; the backstop folds the oldest ones.
        folds_seen = 0
        result = None
        for name in APPEND_ORDER:
            session.append_span(spans[name].span_id)
            result = session.call()
            new_folds = list(session.state.folds.values())[folds_seen:]
            folds_seen = len(session.state.folds)
            note_text = "".join(
                f"; {f.trigger} fold {f.fold_id} folded {len(f.victims)} segments, freed {f.freed_tokens} tokens" for f in new_folds
            )
            trace.append(f"call {result.call_id}: +{name}, window {window_tokens(session.state, blobs)} tokens{note_text}")
        backstops = [f for f in session.state.folds.values() if f.trigger == "backstop"]

        # Ask the three questions.
        questions = [
            Question("Q1", "What is DEFAULT_TIMEOUT?", spans["DEFAULT_TIMEOUT"].span_id, "37"),
            Question("Q2", "What is the maximum backoff delay in seconds?", spans["backoff_delay"].span_id, "300"),
            Question("Q3", "What is MAX_CONNECTIONS?", None, "unknown"),
        ]
        verdicts: dict[str, str] = {}
        refetches: list[dict] = []
        for q in questions:
            answer, verdict = ask(session, model, result, q)
            if verdict is None:
                verdicts[q.qid] = "abstain"
                trace.append(f"{q.qid} {q.question} -> model abstains")
                continue
            trace.append(f"{q.qid} {q.question} -> {answer.kind} {answer.text!r}: tier {verdict.tier} {verdict.action} ({verdict.reason})")
            if verdict.action == "accept":
                verdicts[q.qid] = "accept"
                continue
            if verdict.action != "refetch" or not verdict.support:
                verdicts[q.qid] = verdict.action
                continue
            # The verifier demands a re-fetch: bring the spans back verbatim and ask again.
            for sid in verdict.support:
                res = session.refetch(sid)
                name = label(session, session.state.spans[sid])
                refetches.append({"span": name, "status": res.status})
                trace.append(f"   refetch {name}: {res.status}")
            result = session.call()
            answer, verdict = ask(session, model, result, q)
            if verdict is None:
                verdicts[q.qid] = "abstain-after-refetch"
                continue
            trace.append(f"   retry -> {answer.kind} {answer.text!r}: tier {verdict.tier} {verdict.action} ({verdict.reason})")
            verdicts[q.qid] = "accept-after-refetch" if verdict.action == "accept" else f"{verdict.action}-after-refetch"

        # Edit the module: the constant's span goes stale, the reformatted function still resolves.
        session.update_source(module.source_id, edited_module())
        resolutions = {}
        for name in ["DEFAULT_TIMEOUT", "backoff_delay"]:
            res = session.refetch(spans[name].span_id)
            resolutions[name] = res.status
            refetches.append({"span": label(session, spans[name]), "status": res.status})
            trace.append(f"source edited; refetch {label(session, spans[name])}: {res.status} {res.note}".rstrip())
        final = session.call()
        stale_spans = sorted(label(session, session.state.spans[sid]) for sid, k in final.read_map.items() if k == "stale")
        trace.append(f"call {final.call_id}: {len(stale_spans)} stale spans, e.g. {stale_spans[:2]}")

        # Verify the untouched ledger, then two tampered copies.
        events = ledger.events()
        violations = [f"{v.prop} seq={v.seq}: {v.detail}" for v in check(events, blobs)]
        removed = check(tamper_remove_pinned(events), blobs)
        edited = check(tamper_edit_payload(events), blobs)
        tampered_props = sorted({v.prop for v in removed} | {v.prop for v in edited})
        trace.append(f"check: {len(events)} events, {len(violations)} violations")
        trace.append(f"tamper (evict pinned system segment, rewrite manifest, re-chain): {sorted({v.prop for v in removed})}")
        trace.append(f"tamper (edit a claim without re-chaining): {sorted({v.prop for v in edited})}")

    ok = (
        bool(backstops)
        and not violations
        and {"P7", "P11"} <= set(tampered_props)
        and verdicts == {"Q1": "accept", "Q2": "accept-after-refetch", "Q3": "abstain"}
        and bool(refetches)
        and resolutions["DEFAULT_TIMEOUT"] == "STALE"
        and resolutions["backoff_delay"] in ("OK", "MOVED")
        and f"{MODULE_PATH}#DEFAULT_TIMEOUT" in stale_spans
    )
    trace.append(f"ok: {ok}")
    sys.stdout.write("\n".join(trace) + "\n")
    return {
        "ok": ok,
        "violations": violations,
        "tampered_props": tampered_props,
        "refetches": refetches,
        "verdicts": verdicts,
        "stale_spans": stale_spans,
    }


if __name__ == "__main__":
    raise SystemExit(0 if main()["ok"] else 1)
