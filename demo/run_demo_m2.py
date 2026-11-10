"""Milestone 2 end-to-end demo: a scripted model behind the real Anthropic SDK, fully offline.

What it does, in plain words:
  1. Opens a fresh ledger (in a temporary folder) with a payoff-gated backstop, Haiku 5.5
     prices from pricing/anthropic-2026-10.json, a small context budget and a pinned system
     prompt, and registers two embedded Python modules as spans.
  2. Runs question-and-answer turns through the real `anthropic` SDK client. The client's HTTP
     transport is a FakeAnthropicServer (httpx2.MockTransport), so no network is used and no
     key exists: the only key is the literal "test-key-not-real". The fake server simulates
     the prompt cache and reports usage like the API does.
  3. Turn 1: the model searches, re-fetches spans and answers with a verbatim quote; "37"
     matches DEFAULT_TIMEOUT = 37.0 through numeric normalization.
  4. Reading turns push the window past the hard threshold. The backstop first refuses to fold
     because the fold would not pay back (J <= 0), then folds once it does.
  5. The model tries a bloated fold of its own and is refused (fold_would_grow_window); its
     tool input alone is too big to answer within the budget, so the turn abstains.
  6. A question whose evidence now exists only as a summary: the model cites the summary, the
     verifier routes it to ctx_refetch, and the re-answered claim is grounded.
  7. A question with no evidence: the model abstains.
  8. The source changes: re-fetching shows STALE for an edited constant and MOVED for a moved
     function.
  9. Prints the cost report (per call, totals, cache hit ratio, fold saving predicted vs an
     estimate of what it realized), checks every invariant (expects none violated) and shows
     that two tampered copies of the ledger are caught (P12 and P16).

How to run:  .venv/bin/python demo/run_demo_m2.py   (exit code 0 means every expectation held)
What it changes: nothing outside a temporary folder that is deleted at the end (unless you
call main(db_path=...) to keep the ledger file, e.g. for `context-ledger cost <db>`).
Read-only with respect to everything else: no network, no API key, no paid calls.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "src")]

import anthropic  # noqa: E402

from context_ledger.anthropic.client import LedgerClient  # noqa: E402
from context_ledger.anthropic.fake import FakeAnthropicServer  # noqa: E402
from context_ledger.cost import cost_report  # noqa: E402
from context_ledger.fold import BackstopPolicyV1  # noqa: E402
from context_ledger.invariants import check  # noqa: E402
from context_ledger.pricing import PricingTable, usd  # noqa: E402
from context_ledger.prompts import SYSTEM_PROMPT, answer_text  # noqa: E402
from context_ledger.render import rendered_window_tokens  # noqa: E402
from context_ledger.session import Session  # noqa: E402
from context_ledger.store import Ledger, SqliteBlobStore, rechain  # noqa: E402
from context_ledger.tools import dispatch  # noqa: E402

MODEL = "claude-haiku-5-5"
BUDGET = 4300
PRICING = ROOT / "pricing" / "anthropic-2026-10.json"
CLIENT_PATH = "inventory/client.py"
CLIENT = '''\
"""HTTP client settings and retry helpers for the inventory service."""

import functools
import time

DEFAULT_TIMEOUT = 37.0
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
POLICY_PATH = "inventory/policy.py"
POLICY = '''\
"""Restocking policy for the inventory service."""

SAFETY_STOCK_DAYS = 14
REORDER_BATCH = 48
SUPPLIER_LEAD_DAYS = {"A": 3, "B": 7, "C": 21}


def daily_demand(history):
    """Average units sold per day over the last 28 days of `history`."""
    recent = history[-28:]
    return sum(recent) / max(len(recent), 1)


def safety_stock(sku, history):
    """Units held back for demand spikes: SAFETY_STOCK_DAYS of average demand."""
    return round(daily_demand(history) * SAFETY_STOCK_DAYS)


def reorder_point(sku, history):
    """Reorder when stock falls to lead-time demand plus safety stock."""
    lead = SUPPLIER_LEAD_DAYS[sku[0]]
    return round(daily_demand(history) * lead) + safety_stock(sku, history)


def order_quantity(sku, on_hand, history):
    """Whole batches of REORDER_BATCH units that lift stock above the reorder point."""
    gap = reorder_point(sku, history) - on_hand
    batches = max(0, -(-gap // REORDER_BATCH))
    return batches * REORDER_BATCH


class RestockPlanner:
    """Plans purchase orders for one warehouse."""

    def __init__(self, warehouse, max_lines=200):
        self.warehouse = warehouse
        self.max_lines = max_lines

    def plan(self, stock, histories):
        """One order line per SKU that needs stock, largest first, at most max_lines."""
        lines = [(sku, order_quantity(sku, stock[sku], histories[sku])) for sku in stock]
        lines = [line for line in lines if line[1] > 0]
        lines.sort(key=lambda line: -line[1])
        return lines[: self.max_lines]

    def approve(self, lines, budget_units=5000):
        """Approve lines in order until the unit budget is spent."""
        approved, used = [], 0
        for sku, qty in lines:
            if used + qty > budget_units:
                break
            approved.append((sku, qty))
            used += qty
        return approved
'''

# What the scripted model sees: span headers, search hits and fold lines in the request.
HEADER = re.compile(r"\[(@[0-9A-Z]+) ([^\]\s]+)\]")
HIT = re.compile(r"^(@[0-9A-Z]+) (\S+) \(", re.MULTILINE)
FOLD_LINE = re.compile(r"^- \((@[0-9A-Z]+)\) (.*) \[([^\]]*)\]$", re.MULTILINE)


def request_text(request: dict) -> str:
    """All text the request carries, in order (text blocks and tool_result contents)."""
    parts = []
    for message in request["messages"]:
        for block in message["content"]:
            if block["type"] == "text":
                parts.append(block["text"])
            elif block["type"] == "tool_result":
                parts += [c["text"] for c in block["content"]]
    return "\n".join(parts)


def last_text(request: dict) -> str:
    block = request["messages"][-1]["content"][-1]
    return block["text"] if block["type"] == "text" else "\n".join(c["text"] for c in block["content"])


def verbatim_alias(request: dict, label: str) -> str | None:
    """The alias the request shows on a verbatim span whose label ends with `label`."""
    found = [a for a, where in HEADER.findall(request_text(request)) if where.endswith("#" + label)]
    return found[-1] if found else None


def respond(*content: dict) -> dict:
    stop = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
    return {"content": list(content), "stop_reason": stop}


def call_tool(name: str, n: int, **arguments) -> dict:
    return {"type": "tool_use", "id": f"toolu_{name}_{n:03d}", "name": name, "input": arguments}


class ScriptedModel:
    """A deterministic stand-in for the model. Each method returns a callable of the request
    (as the FakeAnthropicServer wants) that decides from the request text alone."""

    def __init__(self) -> None:
        self.n = 0
        self.aliases: dict[str, str] = {}
        self.cited_summary = ""

    def _id(self) -> int:
        self.n += 1
        return self.n

    def search(self, query: str):
        return lambda request: respond(call_tool("ctx_search", self._id(), query=query))

    def refetch_hits(self, *labels: str):
        """Re-fetch the search hits (from the latest tool result) whose labels end with `labels`;
        the model remembers each alias, as it would from its earlier context."""

        def act(request):
            hits = HIT.findall(last_text(request))
            refs = [a for a, where in hits if any(where.endswith("#" + label) for label in labels)]
            self.aliases.update({where.split("#")[1]: a for a, where in hits})
            return respond(call_tool("ctx_refetch", self._id(), refs=refs))

        return act

    def answer(self, *claims: tuple[str, str, str]):
        """Answer with (text, label, quote) claims, citing the verbatim span by its alias."""

        def act(request):
            body = [{"text": t, "citations": [{"ref": verbatim_alias(request, label), "quote": q}]} for t, label, q in claims]
            return respond({"type": "text", "text": answer_text(body)})

        return act

    def cite_summary(self, text: str, mentions: str):
        """Answer from the summary line about `mentions`, or, if that was folded again, from the
        "earlier fold" line that now stands for it: a guess from a summary."""

        def act(request):
            lines = FOLD_LINE.findall(request_text(request))
            alias, line, _ = next((m for m in lines if mentions in m[1]), None) or next(m for m in lines if m[1].startswith("earlier fold"))
            self.cited_summary = alias
            body = [{"text": text, "citations": [{"ref": alias, "quote": line}]}]
            return respond({"type": "text", "text": answer_text(body)})

        return act

    def refetch_cited_summary(self):
        """After the verification message, re-fetch what the cited summary line stands for."""
        return lambda request: respond(call_tool("ctx_refetch", self._id(), refs=[self.cited_summary]))

    def refetch_labels(self, *labels: str):
        """Re-fetch spans by alias: from a verbatim header, else from the fold line about them."""

        def act(request):
            text = request_text(request)
            refs = []
            for label in labels:
                cited = [cites.split()[0] for _, line, cites in FOLD_LINE.findall(text) if f"#{label}:" in line]
                refs.append(verbatim_alias(request, label) or (cited[-1] if cited else self.aliases[label]))
            return respond(call_tool("ctx_refetch", self._id(), refs=refs))

        return act

    def status(self):
        return lambda request: respond(call_tool("ctx_status", self._id()))

    def bloated_fold(self):
        """Fold the run ctx_status advises first, with a summary (32 sentences of 400 characters,
        the most the schema allows) far longer than the run."""

        def act(request):
            run = re.search(r"^  1\. ((?:@[0-9A-Z]+ ?)+):", last_text(request), re.MULTILINE).group(1).split()
            sentence = ("Everything in these segments, restated at length for safety. " * 7)[:400]
            summary = [{"text": sentence, "cites": []}] * 32
            return respond(call_tool("ctx_fold", self._id(), targets=run, summary=summary))

        return act

    def abstain(self):
        return lambda request: respond({"type": "text", "text": answer_text([], abstain=True)})


class DemoSession(Session):
    """A Session that keeps every backstop analysis, so the demo can show its decisions."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.log: list[tuple] = []
        self.after: dict[int, int] = {}  # call number -> rendered window tokens after the backstop

    def prepare_call(self):
        used = rendered_window_tokens(self.state, self.blobs)
        fold = super().prepare_call()
        self.log.append((len(self.state.calls) + 1, used, self.last_backstop, fold))
        self.after[len(self.state.calls) + 1] = rendered_window_tokens(self.state, self.blobs)
        return fold


def edited_policy() -> str:
    """REORDER_BATCH changes value; reorder_point moves down (a comment block is inserted above
    it) but its code is unchanged."""
    return POLICY.replace("REORDER_BATCH = 48", "REORDER_BATCH = 60").replace(
        "def reorder_point(", "# Lead times come from the supplier contracts.\n# Reviewed quarterly.\n\n\ndef reorder_point("
    )


def tamper_freed_tokens(events: list) -> list:
    """Copy of the ledger with the backstop fold's freed_tokens changed by one, re-chained."""
    events = list(events)
    i = next(i for i, e in enumerate(events) if e.type == "fold" and e.payload["fold"]["trigger"] == "backstop")
    fold = {**events[i].payload["fold"], "freed_tokens": events[i].payload["fold"]["freed_tokens"] + 1}
    events[i] = replace(events[i], payload={"fold": fold})
    return rechain(events)


def tamper_cost(events: list) -> list:
    """Copy of the ledger with one model call's cost off by one pico-USD, re-chained."""
    events = list(events)
    i = next(i for i, e in enumerate(events) if e.type == "model_call")
    events[i] = replace(events[i], payload={**events[i].payload, "cost_pusd": events[i].payload["cost_pusd"] + 1})
    return rechain(events)


def main(db_path: str | Path | None = None) -> dict:
    """Run the demo, print the story and return its numbers (see the module docstring)."""
    if db_path is not None:
        return _run(Path(db_path))
    with tempfile.TemporaryDirectory() as tmp:
        return _run(Path(tmp) / "demo_m2.sqlite")


def _run(db: Path) -> dict:
    def say(line: str) -> None:
        sys.stdout.write(line + "\n")

    with Ledger(db) as ledger, SqliteBlobStore(db) as blobs:
        table = PricingTable.load(PRICING)
        policy = BackstopPolicyV1(budget_tokens=BUDGET, model=MODEL, ttl="5m")
        session = DemoSession(ledger, blobs, BUDGET, policy, pricing=table)
        session.append_raw("system", SYSTEM_PROMPT, pinned=True)
        client_src = session.register_source(CLIENT_PATH, CLIENT, media_type="text/x-python")
        policy_src = session.register_source(POLICY_PATH, POLICY, media_type="text/x-python")
        n_spans = len(session.register_python_spans(client_src, CLIENT_PATH)) + len(session.register_python_spans(policy_src, POLICY_PATH))
        say(f"[setup] {n_spans} Python spans in {CLIENT_PATH} and {POLICY_PATH}; budget {BUDGET} rendered tokens; "
            f"model {MODEL}; prices {table.version} (verified={table.verified}); backstop-v1, lindy R-hat, ttl 5m")  # fmt: skip

        server = FakeAnthropicServer([], pricing=table)
        sdk = anthropic.Anthropic(
            api_key="test-key-not-real", base_url="http://anthropic.test", max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=server.as_sync_transport()),
        )  # fmt: skip
        tool_log: list[tuple] = []  # (tool, outcome, events the handler itself appended)

        def recording_dispatch(sess, name, arguments, **kw):
            before = len(sess.events)
            outcome = dispatch(sess, name, arguments, **kw)
            tool_log.append((name, outcome, sess.events[before:]))
            return outcome

        llm = LedgerClient(session, sdk, model=MODEL, max_tokens=1024, dispatch=recording_dispatch)
        model = ScriptedModel()
        answers: dict[str, dict] = {}
        logged = 0

        def turn(tag: str, question: str, *script) -> object:
            nonlocal logged
            server.script.extend(script)
            result = llm.run_turn(question, turn_id=tag)
            answers[tag] = {
                "question": question,
                "reason": result.reason,
                "statuses": [c.status for c in result.claims],
                "emitted": len(result.emitted),
                "refetch_routes": result.refetch_routes,
                "cost_usd": usd(result.cost_pusd),
            }
            window = rendered_window_tokens(session.state, session.blobs)
            say(f"[{tag}] {question} -> {result.reason}; claims {answers[tag]['statuses']}; "
                f"{len(result.steps)} calls, {usd(result.cost_pusd)}; window now {window} tokens")  # fmt: skip
            for call, used, analysis, fold in session.log[logged:]:
                if fold is not None:
                    d, decision = fold.policy, analysis.decision
                    say(f"    backstop before call {call}: FOLD ({d['zone']}, cache {'cold' if d['cache_cold'] else 'warm'}) "
                        f"[{d['i']},{d['j']}) {len(fold.victims)} segments: L {d['L']} -> {d['L'] - d['delta']} "
                        f"(delta {d['delta']}), R* {float(decision.rstar):.2f}, R-hat {d['rhat']}, J {usd(decision.J)}"
                        + (" (mandatory zone: payback waived)" if d["zone"] == "mandatory" else ""))  # fmt: skip
                elif analysis.zone in ("discretionary", "mandatory"):
                    best = analysis.runs[0] if analysis.runs else None
                    detail = (
                        f"best [{best.i},{best.j}) delta {best.delta}, R* {float(best.rstar):.2f}, R-hat {analysis.rhat}, J {usd(best.J)}"
                        if best else "no feasible run"
                    )  # fmt: skip
                    say(f"    backstop before call {call}: no fold ({analysis.zone}, cache {'cold' if analysis.cache_cold else 'warm'}, "
                        f"window {used}): {detail}, J <= 0 so the fold would not pay back")  # fmt: skip
            logged = len(session.log)
            return result

        # Turn 1: search, re-fetch, answer with a numeric-normalized quote ("37" vs 37.0).
        turn("Q1", "What is the default HTTP timeout of the inventory client, in seconds?",
             model.search("timeout"),
             model.refetch_hits("DEFAULT_TIMEOUT", "InventoryClient.__init__", "InventoryClient.fetch"),
             model.answer(("The default timeout is 37 seconds.", "DEFAULT_TIMEOUT", "DEFAULT_TIMEOUT = 37.0")))  # fmt: skip

        # Reading turns: each brings more code in, until the backstop has to weigh a fold.
        reading = [
            ("Q2", "Which exception makes the client retry?", "retry backoff", ["retrying", "backoff_delay"],
             ("The client retries on ConnectionError.", "retrying", "except ConnectionError:")),
            ("Q3", "How many units are in one reorder batch?", "batch", ["REORDER_BATCH", "order_quantity"],
             ("One reorder batch holds 48 units.", "REORDER_BATCH", "REORDER_BATCH = 48")),
            ("Q4", "How many days of safety stock are held?", "safety stock", ["SAFETY_STOCK_DAYS", "safety_stock"],
             ("Safety stock covers 14 days of demand.", "SAFETY_STOCK_DAYS", "SAFETY_STOCK_DAYS = 14")),
            ("Q5", "What is the default unit budget when approving orders?", "approve budget", ["RestockPlanner.approve"],
             ("Approval stops at a default budget of 5000 units.", "RestockPlanner.approve", "def approve(self, lines, budget_units=5000):")),
            ("Q6", "How many order lines can a plan have by default?", "plan lines", ["RestockPlanner.plan", "RestockPlanner.__init__"],
             ("A plan has at most 200 lines by default.", "RestockPlanner.__init__", "def __init__(self, warehouse, max_lines=200):")),
            ("Q7", "How long is supplier C's lead time?", "lead days", ["SUPPLIER_LEAD_DAYS", "reorder_point", "daily_demand"],
             ("Supplier C has a lead time of 21 days.", "SUPPLIER_LEAD_DAYS", 'SUPPLIER_LEAD_DAYS = {"A": 3, "B": 7, "C": 21}')),
            ("Q8", "Which user agent does the client send?", "user agent", ["USER_AGENT"],
             ("The client identifies itself as inventory-client/2.1.", "USER_AGENT", 'USER_AGENT = "inventory-client/2.1"')),
            ("Q9", "How many times does the client retry?", "retries", ["MAX_RETRIES"],
             ("The client retries 4 times.", "MAX_RETRIES", "MAX_RETRIES = 4")),
            ("Q10", "How is an item URL built?", "url sku", ["InventoryClient.url_for"],
             ("Item URLs end with the SKU after the items path.", "InventoryClient.url_for", 'return f"{self.base_url}/items/{sku}"')),
            ("R1", "Which SKU does the health check fetch?", "health ping", ["ping"],
             ('The health check fetches the "health" SKU.', "ping", 'return client.fetch("health")')),
            ("R2", "Below how many units do A-SKUs get restocked?", "restock threshold", ["InventoryClient.restock_threshold"],
             ("SKUs starting with A are restocked below 12 units.", "InventoryClient.restock_threshold", 'return 12 if sku.startswith("A") else 25')),
            ("R3", "How many days of history does demand use?", "demand history", ["daily_demand"],
             ("Demand averages the last 28 days.", "daily_demand", "Average units sold per day over the last 28 days")),
            ("R4", "What does fetch return for a SKU?", "fetch sku", ["InventoryClient.fetch", "InventoryClient"],
             ("fetch returns the SKU, its URL and the timeout.", "InventoryClient.fetch", 'return {"sku": sku, "url": self.url_for(sku), "timeout": self.timeout}')),
            ("R5", "How are order lines sorted in a plan?", "plan lines", ["RestockPlanner.plan"],
             ("Plan lines are sorted largest quantity first.", "RestockPlanner.plan", "lines.sort(key=lambda line: -line[1])")),
        ]  # fmt: skip
        for tag, question, query, labels, claim in reading:
            turn(tag, question, model.search(query), model.refetch_hits(*labels), model.answer(claim))

        # The model tries to fold with a summary longer than what it folds. Its tool input is now
        # the newest tool pair, bigger than what the backstop can free: the harness abstains
        # (budget_exceeded) rather than send a request above the mandatory zone (BUILD-V2b).
        bloat = turn("Q11", "Check the context budget, tidy up, then tell me how plan lines are sorted.",
                     model.status(),
                     model.bloated_fold())  # fmt: skip
        name, outcome, appended = next(entry for entry in tool_log if entry[0] == "ctx_fold")
        fold_error_text = outcome.text
        bloated_refused = outcome.is_error and fold_error_text.startswith("error: fold_would_grow_window") and not appended
        say(f"    ctx_fold -> {fold_error_text}: refused; the tool call recorded {len(appended)} events, only its result was added. "
            "The bloated tool input itself pushed the window into the mandatory zone, and as the newest tool pair it cannot be folded "
            "this turn: the answer call would exceed the budget, so the turn abstains.")  # fmt: skip

        # Evidence that now lives only in a summary: cite it, get routed to ctx_refetch, re-answer.
        summary_turn = turn("Q12", "What is the longest wait between retries, in seconds?",
                            model.cite_summary("The longest wait between retries is 300 seconds.", "backoff_delay"),
                            model.refetch_cited_summary(),
                            model.answer(("The longest wait between retries is 300 seconds.", "backoff_delay", "return min(2 ** attempt, 300)")))  # fmt: skip

        # No evidence anywhere: abstain.
        turn("Q13", "What is MAX_CONNECTIONS?", model.search("max connections"), model.abstain())

        # The policy module changes: an edited constant goes STALE, a moved function is MOVED.
        session.update_source(policy_src.source_id, edited_policy())
        edit = turn("Q14", "The policy file changed. Re-check the batch size and the reorder point.",
                    model.refetch_labels("REORDER_BATCH", "reorder_point"),
                    model.answer(("The reorder point adds safety stock to lead-time demand.", "reorder_point",
                                  "return round(daily_demand(history) * lead) + safety_stock(sku, history)")))  # fmt: skip
        refetch_text = blobs.get(next(
            e.payload["segment"]["text_hash"] for e in reversed(session.events)
            if e.type == "segment_appended" and e.payload["segment"].get("tool_use_id", "").startswith("toolu_ctx_refetch")
            and e.payload["segment"]["ref_kind"] == "raw"
        )).decode()  # fmt: skip
        say("    ctx_refetch after the edit -> " + " | ".join(refetch_text.splitlines()))

        # Every backstop decision of the run: refusals (J <= 0) and folds.
        rejected = [
            {"call": call, "zone": a.zone, "used": used, "i": a.runs[0].i, "j": a.runs[0].j, "delta": a.runs[0].delta,
             "rhat": a.rhat, "rstar": str(a.runs[0].rstar), "J_usd": usd(a.runs[0].J), "cache_cold": a.cache_cold}
            for call, used, a, fold in session.log if fold is None and a.zone == "discretionary" and a.runs
        ]  # fmt: skip
        decisions, window_before_after = [], []
        for call, used, a, fold in session.log:
            if fold is not None:
                d = fold.policy
                decisions.append({"call": call, "zone": d["zone"], "L": d["L"], "P_i": d["P_i"], "i": d["i"], "j": d["j"],
                                  "delta": d["delta"], "rhat": d["rhat"], "cache_cold": d["cache_cold"], "J": d["J"],
                                  "rstar": d["rstar"], "J_usd": usd(a.decision.J), "victims": len(fold.victims)})  # fmt: skip
                window_before_after.append([used, session.after[call]])

        # Cost report, invariants, tampering.
        events = ledger.events()
        report = cost_report(events)
        say("[cost]\n" + report.render())
        violations = [f"{v.prop} seq={v.seq}: {v.detail}" for v in check(events, blobs)]
        tamper = {
            "freed_tokens+1": sorted({v.prop for v in check(tamper_freed_tokens(events), blobs)}),
            "cost_pusd+1": sorted({v.prop for v in check(tamper_cost(events), blobs)}),
        }
        say(f"[check] {len(events)} events, {len(violations)} violations; tampered copies: {tamper}")
        calls = [
            {"call_id": e.payload["call_id"], "stop_reason": e.payload["stop_reason"], "cost_pusd": e.payload["cost_pusd"],
             "breakpoints": len(e.payload["breakpoints"]), **{k: e.payload["usage"][k] for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")}}
            for e in events if e.type == "model_call"
        ]  # fmt: skip
        result = {
            "calls": calls,
            "folds": decisions,
            "rejected_folds": rejected,
            "window_before_after": window_before_after,
            "answers": answers,
            "refetch_routes": summary_turn.refetch_routes,
            "bloated_fold": {"refused": bloated_refused, "text": fold_error_text.splitlines()[0], "turn": bloat.reason},
            "source_edit": refetch_text.splitlines(),
            "cost_report": report.to_dict(),
            "violations": violations,
            "tamper": tamper,
            "events": [e.type for e in events],
            "head": ledger.head(),
        }
    ok = (
        not violations
        and bool(decisions)
        and bool(rejected)
        and rejected[0]["call"] < decisions[0]["call"]
        and all(after < before for before, after in window_before_after)
        and bloated_refused
        and bloat.reason == "budget_exceeded"
        and answers["Q1"]["statuses"] == ["grounded"]
        and answers["Q12"]["statuses"] == ["needs_refetch", "grounded"]
        and answers["Q13"]["reason"] == "model_abstained"
        and any(" STALE" in line for line in refetch_text.splitlines())
        and any(" MOVED " in line for line in refetch_text.splitlines())
        and tamper == {"freed_tokens+1": ["P12"], "cost_pusd+1": ["P16"]}
        and edit.reason == "answered"
    )
    result["ok"] = ok
    say(f"ok: {ok}")
    return result


if __name__ == "__main__":
    raise SystemExit(0 if main()["ok"] else 1)
