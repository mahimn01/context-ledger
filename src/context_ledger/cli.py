"""Command line: check, replay and export a ledger file.

    python -m context_ledger.cli check <db>                 exit 1 if any invariant fails
    python -m context_ledger.cli replay <db> --call <id>    manifest and read map as JSON
    python -m context_ledger.cli export-prov <db>           minimal W3C PROV-JSON
    python -m context_ledger.cli cost <db> [--json]         per-call costs, totals, cache hits, fold savings

(`context-ledger <command> ...` is the same, once the package is installed.)

All commands are read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .cost import cost_report
from .invariants import check
from .model import Claim, Event, FoldRecord
from .store import Ledger, SqliteBlobStore
from .window import ReplayError, WindowState, apply_event, down, fold_sentence_refs, manifest, manifest_hash, read_map, state_at_call


def _out(text: str) -> None:
    sys.stdout.write(text + "\n")


def _open(db: str) -> tuple[Ledger, SqliteBlobStore]:
    if not Path(db).is_file():
        raise FileNotFoundError(f"no ledger file at {db}")
    return Ledger(db), SqliteBlobStore(db)


def export_prov(events: Sequence[Event]) -> dict:
    """Minimal PROV-JSON: spans and blobs are entities, calls and folds are activities; `used`
    links calls to exposed spans, `wasDerivedFrom` links each fold segment to the spans in
    down() of its cites, `wasGeneratedBy` links claims (and fold segments) to their activity."""
    doc: dict[str, dict] = {
        "prefix": {"clp": "urn:clp:"},
        "entity": {},
        "activity": {},
        "used": {},
        "wasDerivedFrom": {},
        "wasGeneratedBy": {},
    }

    def relation(kind: str, body: dict) -> None:
        doc[kind][f"_:{kind}{len(doc[kind]) + 1}"] = body

    def blob(h: str) -> str:
        doc["entity"].setdefault(f"clp:blob/{h}", {"prov:type": "clp:Blob"})
        return f"clp:blob/{h}"

    state = WindowState()
    for event in events:
        apply_event(state, event)
        p = event.payload
        if event.type == "source_registered":
            blob(p["source"]["content_hash"])
        elif event.type == "source_updated":
            blob(p["content_hash"])
        elif event.type == "span_registered":
            span = p["span"]
            doc["entity"][f"clp:span/{span['span_id']}"] = {
                "prov:type": "clp:Span",
                "clp:source": span["source_id"],
                "clp:source_blob": blob(span["source_hash"]),
                "clp:text_blob": blob(span["text_hash"]),
            }
        elif event.type == "segment_appended":
            blob(p["segment"]["text_hash"])
        elif event.type == "call":
            call = f"clp:call/{p['call_id']}"
            doc["activity"][call] = {"prov:type": "clp:Call", "clp:manifest_hash": p["manifest_hash"]}
            for span_id in p["exposures"]:
                relation("used", {"prov:activity": call, "prov:entity": f"clp:span/{span_id}"})
        elif event.type == "fold":
            fold = FoldRecord.from_dict(p["fold"])
            activity, segment = f"clp:fold/{fold.fold_id}", f"clp:segment/{fold.summary_segment_id}"
            doc["activity"][activity] = {"prov:type": "clp:Fold", "clp:trigger": fold.trigger}
            doc["entity"][segment] = {"prov:type": "clp:FoldSegment", "clp:text_blob": blob(fold.summary_hash)}
            relation("wasGeneratedBy", {"prov:entity": segment, "prov:activity": activity})
            for span_id in sorted(set().union(*(down(state, r) for r in fold_sentence_refs(fold)))):
                relation("wasDerivedFrom", {"prov:generatedEntity": segment, "prov:usedEntity": f"clp:span/{span_id}"})
        elif event.type == "claim":
            claim = Claim.from_dict(p["claim"])
            entity = f"clp:claim/{claim.claim_id}"
            doc["entity"][entity] = {"prov:type": "clp:Claim", "clp:status": claim.status, "clp:text": claim.text}
            if claim.call_id is not None:
                relation("wasGeneratedBy", {"prov:entity": entity, "prov:activity": f"clp:call/{claim.call_id}"})
    return doc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m context_ledger.cli", description="Inspect a context-ledger file.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="report invariant violations").add_argument("db")
    replay_cmd = commands.add_parser("replay", help="print one call's manifest and read map")
    replay_cmd.add_argument("db")
    replay_cmd.add_argument("--call", required=True, dest="call_id")
    commands.add_parser("export-prov", help="print W3C PROV-JSON").add_argument("db")
    cost_cmd = commands.add_parser("cost", help="print the cost report (prices from the ledger's price table)")
    cost_cmd.add_argument("db")
    cost_cmd.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    try:
        ledger, blobs = _open(args.db)
    except FileNotFoundError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    with ledger, blobs:
        events = ledger.events()
        if args.command == "check":
            violations = check(events, blobs)
            for v in violations:
                _out(f"{v.prop} seq={v.seq}: {v.detail}")
            _out(f"{len(events)} events, {len(violations)} violations")
            return 1 if violations else 0
        if args.command == "cost":
            report = cost_report(events)
            if args.json:
                _out(json.dumps(report.to_dict(), indent=2, sort_keys=True))
            else:
                _out(report.render())
                _out("Prices come from the price table recorded in the ledger; amounts are estimates if it is unverified.")
            return 0
        try:
            if args.command == "replay":
                state = state_at_call(events, args.call_id)
                body = {
                    "call_id": args.call_id,
                    "manifest_hash": manifest_hash(state),
                    "manifest": manifest(state),
                    "read_map": read_map(state),
                }
            else:
                body = export_prov(events)
        except KeyError:
            sys.stderr.write(f"error: no call {args.call_id} in {args.db}\n")
            return 2
        except ReplayError as exc:
            sys.stderr.write(f"error: ledger does not replay: {exc}\n")
            return 2
        _out(json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
