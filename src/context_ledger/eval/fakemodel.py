"""ScriptedPilotModel: a deterministic stand-in model for offline pilot runs (never live).

It answers each request from what the request shows, the way the arms expect a model to:

- It finds the current question (the last pilot question in the user's text) and its subject.
- If the answer is visible verbatim (a span "[@3 path#name]", a read "[path:a-b]" or a RAG
  chunk), it answers with that value, cites the piece's ref and quotes the line it used.
- If the subject appears only in a summary (an L fold "[fold @A]" or a B4 compaction summary),
  it guesses: a wrong value citing the summary. This simulates the hallucination the pilot
  measures; in L the verifier then routes the claim to ctx_refetch.
- After an L verification message, or a refetch-before-abstaining reminder, it calls
  ctx_refetch on the aliases the message names.
- Otherwise, with tools, it searches once and reads up to two hits; then it abstains.
- A compaction request (B4) gets a summary listing the refs read and the names seen.
- For the v2 multi-hop questions ("Resolve any named constant to its literal value") a default or
  constant that is a name is resolved: it answers the literal of the named upper-case constant
  if a verbatim piece shows it (citing both pieces), else it searches for the name and reads.
- For a session probe ("What is the current <attribute> of service <name>?", BUILD-V2c) it
  answers the value of the latest note on that exact pair that it can see ("[note:<pos>]" for
  B3 and B4, a "[@A note:<pos>#a-b]" span for L), citing it; a folded note's summary line makes
  it guess, as for a folded span; in L it otherwise looks with ctx_search and ctx_refetch; else
  it abstains.
- It answers in the format the system prompt asks for: v2 (an `ANSWER:` line and plain claims
  such as "`mod.f` parameter `p` defaults to `3`") when the prompt shows the `ANSWER: <...>`
  line, else the v1 <answer> JSON block.
- BUILD-V3. It reads a code subject's answer only from text of the subject's own module (a span
  or read whose path is that module; locate questions name no module), and a raises answer
  cites every raise line. After an answer-gate message ("Verification: the question asks
  about ..." or "Verification: your answer ...") it explores again; a stale-note message names
  aliases and ctx_refetch, so it refetches them. Under the forced prompt ("Never write
  UNANSWERABLE"), where it would abstain it guesses a wrong value citing the latest verbatim
  piece, or (for half the subjects, chosen by a checksum of the subject's name, and never right
  after the forced-answer message) writes UNANSWERABLE anyway, so a forced run exercises both.
- BUILD-V4, only when the request offers the address tools (`lookup`; arms B5 and L4): where it
  would search and read, it looks the subject up and reads the first address the lookup gives
  (a lookup with no exact address ends the exploration), then searches (where `search` is
  offered) and reads the hits as before; a second hop is looked up by its bare name. A repair
  message ("Repair ...") is answered like any turn: from the verbatim text in view.

Its results say nothing about any real model: it exists to exercise the harness end to end.
"""

from __future__ import annotations

import json
import re

from ..anthropic.client import FORCED_MESSAGE
from .prompts import COMPACTION_PROMPT, UNANSWERABLE, pilot_answer_text, pilot_answer_text_v2
from .pyscan import module_path

QUESTION = re.compile(
    r"What value does parameter `(?P<rparam>\w+)` of `(?P<rq2>[\w.<>]+)` default to\?"
    r"|What is the default value of parameter `(?P<param>\w+)` of `(?P<dq>[\w.<>]+)`\?"
    r"|What is the value of the module constant `(?P<cq>[\w.]+)`\?"
    r"|Which exception classes does `(?P<rq>[\w.<>]+)` raise directly"
    r"|Which module defines the function `(?P<lq>\w+)`\?"
    r"|What is the current (?P<sattr>[a-z ]+?) of service (?P<ssvc>[\w-]+)\?"
)
SPAN_HEADER = re.compile(r"^\[(@[0-9A-Z]+) ([^\]\n]+)\]$", re.M)
READ_HEADER = re.compile(r"^\[([^\]\n:]+):(\d+)-(\d+)\]$", re.M)
FOLD_HEADER = re.compile(r"^\[fold (@[0-9A-Z]+)\]$", re.M)
NOTE_HEADER = re.compile(r"^\[(note:\d+)\]$", re.M)
FOUND_NOTE = re.compile(r"^(@[0-9A-Z]+) (note:\d+)#", re.M)
FOLD_LINE = re.compile(r"^- \((@[0-9A-Z]+)\) (.*) \[[^\]]*\]$", re.M)
HIT = re.compile(r"^(\S+):(\d+)-(\d+)  (.*)$", re.M)
V4_ADDRESS = re.compile(r"(?:^|\s)(\S+\.\w+:\d+-\d+#[0-9a-f]{8})", re.M)
SUMMARY_HEAD = "Summary of the earlier conversation:"


def _text_blocks(content) -> list[tuple[str, str]]:
    """(kind, text) of every text in a message's content, tool results included."""
    if isinstance(content, str):
        return [("text", content)]
    out = []
    for block in content:
        if block.get("type") == "text":
            out.append(("text", block["text"]))
        elif block.get("type") == "tool_result":
            inner = block.get("content")
            for kind, text in _text_blocks(inner if inner is not None else []):
                out.append(("tool_result", text))
        elif block.get("type") == "tool_use":
            out.append(("tool_use", json.dumps({"name": block["name"], "input": block["input"]}, sort_keys=True)))
    return out


def _pieces(text: str) -> list[dict]:
    """Split a text at span, read and fold headers into pieces {kind, ref, path, body}."""
    marks = []
    for m in SPAN_HEADER.finditer(text):
        marks.append((m.start(), m.end(), "span", m.group(1), m.group(2).split("#", 1)[0]))
    for m in READ_HEADER.finditer(text):
        marks.append((m.start(), m.end(), "read", f"{m.group(1)}:{m.group(2)}-{m.group(3)}", m.group(1)))
    for m in FOLD_HEADER.finditer(text):
        marks.append((m.start(), m.end(), "fold", m.group(1), None))
    for m in NOTE_HEADER.finditer(text):
        marks.append((m.start(), m.end(), "note", m.group(1), m.group(1)))
    marks.sort()
    out = []
    for k, (start, end, kind, ref, path) in enumerate(marks):
        stop = marks[k + 1][0] if k + 1 < len(marks) else len(text)
        out.append({"kind": kind, "ref": ref, "path": path, "body": text[end + 1 : stop]})
    return out


def _def_block(body: str, name: str) -> list[str] | None:
    """The lines of the def of `name` in `body`: its signature and body, by indentation."""
    lines = body.split("\n")
    pattern = re.compile(rf"^(\s*)(?:async\s+)?def\s+{re.escape(name)}\s*\(")
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            indent = len(m.group(1))
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > indent or lines[j].lstrip().startswith(")")):
                j += 1
            return lines[i:j]
    return None


def _signature(block: list[str]) -> list[str]:
    out, depth = [], 0
    for line in block:
        out.append(line)
        depth += line.count("(") - line.count(")")
        if depth <= 0 and line.rstrip().endswith(":"):
            break
    return out


def find_answer(template: str, subject: dict, piece: dict) -> tuple[object, str] | None:
    """(answer, quoted line) if this verbatim piece answers the question, else None."""
    body = piece["body"]
    name = subject["name"]
    if template == "constant":
        m = re.search(rf"^{re.escape(name)}\s*(?::[^=\n]+)?=\s*(.+?)\s*(?:#.*)?$", body, re.M)
        return (m.group(1), m.group(0).strip()) if m else None
    lines = _def_block(body, name)
    if lines is None:
        return None
    if template == "default":
        for line in _signature(lines):
            m = re.search(rf"(?<![\w]){re.escape(subject['param'])}\s*(?::\s*[^=,)]+)?=\s*([^,)\n]+)", line)
            if m:
                return m.group(1).strip(), line.strip()
        return None
    if template == "raises":
        indent = len(lines[0]) - len(lines[0].lstrip())
        names, quotes, skip = [], [], None
        for line in lines[1:]:
            depth = len(line) - len(line.lstrip())
            if skip is not None and line.strip() and depth <= skip:
                skip = None
            if skip is not None:
                continue
            if re.match(r"\s*(async\s+def|def|class)\s", line) and depth > indent:
                skip = depth
                continue
            m = re.match(r"\s*raise\s+([\w.]+)", line)
            if m:
                names.append(m.group(1).rsplit(".", 1)[-1])
                quotes += [line.strip()] if line.strip() not in quotes else []
        return (sorted(set(names)), quotes) if names else None
    path = piece["path"] or ""
    return (module_path(path), lines[0].strip()) if path.endswith(".py") else None


def _of_module(piece: dict, template: str, subject: dict) -> bool:
    """A verbatim piece the subject's answer may come from: any for locate (it names no module),
    else one of the subject's own module (its qualified name starts with the piece's module)."""
    if template == "locate" or "." not in subject["qual"]:
        return True
    path = piece["path"] or ""
    return path.endswith(".py") and subject["qual"].startswith(module_path(path) + ".")


def _is_name(value: object) -> bool:
    """A default or constant written as a (dotted) name rather than a literal."""
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z_][\w.]*", value.strip()) is not None and value.strip() not in ("True", "False", "None")


class ScriptedPilotModel:
    """The responder for FakeAnthropicServer (see the module docstring)."""

    def __init__(self, max_reads: int = 2) -> None:
        self.max_reads = max_reads
        self.tool_calls = 0  # numbers tool_use ids, unique over the whole run
        self.v2 = False  # the answer format of the request being answered
        self.forced = False  # the request's system prompt is the forced condition's
        self.after_forced = False  # the request's last message is the forced-answer message
        self.v4 = False  # the request offers the v4 address tools (BUILD-V4)

    def __call__(self, request: dict) -> dict:
        system = "".join(b.get("text", "") for b in request.get("system", []))
        if system == COMPACTION_PROMPT:
            return self._summarize(request)
        self.v2 = "ANSWER: <" in system or f"ANSWER: {UNANSWERABLE}" in system
        self.forced = "Never write UNANSWERABLE" in system
        messages = request["messages"]
        tools = {t["name"] for t in request.get("tools", [])}
        self.v4 = "lookup" in tools
        self.search_tool = "search" in tools
        question, q_index = self._question(messages)
        if question is None:
            return self._reply(self._format(None, [], True))
        template, subject = question
        last = [t for k, t in _text_blocks(messages[-1]["content"]) if k == "text"]
        pieces = [p for m in messages for _, t in _text_blocks(m["content"]) for p in _pieces(t)]
        self.after_forced = bool(last) and last[-1] == FORCED_MESSAGE
        if last and last[-1].startswith(("Verification: the question asks about", "Verification: your answer")):
            return self._explore(subject, messages, q_index, template, pieces) if tools else self._give_up(template, subject, pieces)
        refs = list(dict.fromkeys(re.findall(r"@[0-9A-Z]+", last[-1])))[:8] if last else []
        if "ctx_refetch" in tools and refs and last[-1].startswith(("Verification:", "Before abstaining:")) and "ctx_refetch" in last[-1]:
            return self._tool("ctx_refetch", {"refs": refs})
        if template == "session":
            return self._session(subject, messages, pieces, tools, q_index)
        for piece in reversed(pieces):  # the most recent verbatim text first
            if piece["kind"] in ("span", "read") and _of_module(piece, template, subject):
                found = find_answer(template, subject, piece)
                if found is not None and subject.get("resolve") and _is_name(found[0]):
                    target = str(found[0]).rsplit(".", 1)[-1]
                    hop = next(((p, f) for p in reversed(pieces) if p["kind"] in ("span", "read")
                                for f in [find_answer("constant", {"name": target}, p)] if f is not None and not _is_name(f[0])), None)  # fmt: skip
                    if hop is None:
                        return self._explore({"name": target, "qual": target, "query": target}, messages, q_index)
                    return self._answer(template, subject, (hop[1][0], found[1]), piece["ref"], extra=(hop[0]["ref"], hop[1][1]))
                if found is not None:
                    return self._answer(template, subject, found, piece["ref"])
        guess = self._guess(template, subject, messages, pieces)
        if guess is not None:
            return guess
        if not tools:
            return self._give_up(template, subject, pieces)
        return self._explore(subject, messages, q_index, template, pieces)

    @staticmethod
    def _question(messages: list) -> tuple[tuple[str, dict] | None, int]:
        found, index = None, -1
        for n, message in enumerate(messages):
            if message["role"] != "user":
                continue
            for kind, text in _text_blocks(message["content"]):
                if kind != "text":
                    continue
                for m in QUESTION.finditer(text):
                    resolve = "Resolve any named constant" in text[m.end() :]
                    if m.group("rq2"):
                        found = ("default", {"name": m.group("rq2").rsplit(".", 1)[-1], "param": m.group("rparam"), "qual": m.group("rq2"), "resolve": True})
                    elif m.group("dq"):
                        found = ("default", {"name": m.group("dq").rsplit(".", 1)[-1], "param": m.group("param"), "qual": m.group("dq")})
                    elif m.group("cq"):
                        found = ("constant", {"name": m.group("cq").rsplit(".", 1)[-1], "qual": m.group("cq"), "resolve": resolve})
                    elif m.group("ssvc"):
                        found = ("session", {"name": m.group("ssvc"), "attr": m.group("sattr"), "qual": m.group("ssvc")})
                    elif m.group("rq"):
                        found = ("raises", {"name": m.group("rq").rsplit(".", 1)[-1], "qual": m.group("rq")})
                    else:
                        found = ("locate", {"name": m.group("lq"), "qual": m.group("lq")})
                    index = n
        return found, index

    def _session(self, subject: dict, messages: list, pieces: list[dict], tools: set[str], q_index: int) -> dict:
        """A session probe: the latest visible note on the exact pair, else a guess from a
        folded note's line, else (L) ctx_search then ctx_refetch of the notes it finds, else
        abstain."""
        line = re.compile(rf"^.*\bthe {re.escape(subject['attr'])} of service {re.escape(subject['name'])} is (?:now )?(\S+?)\.$", re.M)
        seen = [(int(p["path"].split(":", 1)[1]), p["ref"], m) for p in pieces if p["kind"] in ("note", "span") and (p["path"] or "").startswith("note:")
                for m in [line.search(p["body"])] if m]  # fmt: skip
        if seen:
            _, ref, m = max(seen, key=lambda s: s[0])
            return self._answer("session", subject, (m.group(1), m.group(0).strip()), ref)
        guess = self._guess("session", subject, messages, pieces)
        if guess is not None or "ctx_search" not in tools:
            return guess or self._give_up("session", subject, pieces)
        since = messages[q_index + 1 :] if q_index >= 0 else []
        uses = [json.loads(t) for m in since if m["role"] == "assistant" for k, t in _text_blocks(m["content"]) if k == "tool_use"]
        if not any(u["name"] == "ctx_search" for u in uses):
            return self._tool("ctx_search", {"query": f"{subject['name']} {subject['attr']}", "limit": 5})
        found = [m.group(1) for m in since if m["role"] == "user" for k, t in _text_blocks(m["content"]) if k == "tool_result" for m in FOUND_NOTE.finditer(t)]
        if found and not any(u["name"] == "ctx_refetch" for u in uses):
            return self._tool("ctx_refetch", {"refs": list(dict.fromkeys(found))[:4]})
        return self._give_up("session", subject, pieces)

    def _guess(self, template: str, subject: dict, messages: list, pieces: list[dict]) -> dict | None:
        """The simulated hallucination: the subject is named only in a summary."""
        wrong = {"default": "0", "constant": "0", "raises": ["Exception"], "locate": "unknown.module", "session": "0"}[template]
        for piece in reversed(pieces):
            if piece["kind"] == "fold":
                for m in FOLD_LINE.finditer(piece["body"]):
                    if "#" in m.group(2).split(": ", 1)[0] and subject["name"] in m.group(2):  # a folded span's line
                        claim = {"text": f"From the summary, the answer is {wrong if not isinstance(wrong, list) else wrong[0]}.", "citations": [{"ref": m.group(1), "quote": None}]}
                        return self._reply(self._format(wrong, [claim]))
        for message in messages:
            for kind, text in _text_blocks(message["content"]):
                if kind == "text" and text.startswith(SUMMARY_HEAD) and subject["name"] in text:
                    ref = next((f"{m.group(1)}:{m.group(2)}-{m.group(3)}" for m in re.finditer(r"(\S+\.py):(\d+)-(\d+)", text)), "summary")
                    claim = {"text": "From the summary of earlier work.", "citations": [{"ref": ref, "quote": text.split("\n", 2)[1][:80] if "\n" in text else text[:80]}]}
                    return self._reply(self._format(wrong, [claim]))
        return None

    def _give_up(self, template: str, subject: dict, pieces: list[dict]) -> dict:
        """Abstain; under the forced prompt guess instead (module docstring): a wrong value whose
        claim cites the latest verbatim piece with its first line as the quote."""
        if not self.forced or (not self.after_forced and sum(map(ord, subject["qual"])) % 2):
            return self._reply(self._format(None, [], True))
        wrong = {"default": "0", "constant": "0", "raises": "Exception", "locate": "unknown.module", "session": "0"}[template]
        piece = next((p for p in reversed(pieces) if p["kind"] in ("span", "read", "note") and p["body"].strip()), None)
        if piece is None:
            citations = [{"ref": "inference", "quote": None}]
        else:
            body = piece["body"].strip()
            line = next((ln.strip() for ln in body.split("\n") if len(ln.strip()) >= 20), body)
            citations = [{"ref": piece["ref"], "quote": line}]
        return self._reply(self._format(wrong, [{"text": f"Best guess: the answer is {wrong}.", "citations": citations}]))

    def _explore(self, subject: dict, messages: list, q_index: int, template: str = "constant", pieces: list[dict] | None = None) -> dict:
        if self.v4:
            return self._explore_v4(subject, messages, q_index, template, pieces)
        since = messages[q_index + 1 :] if q_index >= 0 else []
        uses = [json.loads(t) for m in since if m["role"] == "assistant" for k, t in _text_blocks(m["content"]) if k == "tool_use"]
        searched = [u for u in uses if u["name"] == "corpus_search"]
        reads = [u for u in uses if u["name"] == "corpus_read"]
        if "query" in subject and subject["query"] not in {u["input"]["query"] for u in searched}:  # a second hop to look up
            return self._tool("corpus_search", {"query": subject["query"], "limit": 5})
        if not searched:
            module = subject["qual"].rsplit(".", 1)[0].rsplit(".", 1)[-1] if "." in subject["qual"] else ""
            return self._tool("corpus_search", {"query": f"{subject['name']} {module}".strip(), "limit": 5})
        if len(reads) < self.max_reads + ("query" in subject):
            results = [t for m in since if m["role"] == "user" for k, t in _text_blocks(m["content"]) if k == "tool_result"]
            done = {(r["input"]["path"], r["input"]["start_line"]) for r in reads}
            hits = [(h.group(1), int(h.group(2)), int(h.group(3)), h.group(4)) for t in results for h in HIT.finditer(t)]
            hits.sort(key=lambda h: subject["name"] not in h[3])  # a hit naming the subject first
            for path, start, end, _ in hits:
                if (path, start) not in done:
                    return self._tool("corpus_read", {"path": path, "start_line": start, "end_line": end})
        return self._give_up(template, subject, pieces or [])

    def _explore_v4(self, subject: dict, messages: list, q_index: int, template: str, pieces: list[dict] | None) -> dict:
        """The v4 exploration (module docstring): lookup, read its first address, then search."""
        since = messages[q_index + 1 :] if q_index >= 0 else []
        uses = [json.loads(t) for m in since if m["role"] == "assistant" for k, t in _text_blocks(m["content"]) if k == "tool_use"]
        results = [t for m in since if m["role"] == "user" for k, t in _text_blocks(m["content"]) if k == "tool_result"]
        name = subject.get("query") or subject["qual"]
        looked = [u["input"]["name"] for u in uses if u["name"] == "lookup"]
        if name not in looked:
            return self._tool("lookup", {"name": name})
        read = {u["input"]["address"] for u in uses if u["name"] == "read"}
        exact = [m.group(1) for t in results if not t.startswith("no exact address") for m in V4_ADDRESS.finditer(t)]
        for address in exact:
            if address not in read and len(read) < self.max_reads + ("query" in subject):
                return self._tool("read", {"address": address})
        searched = [u for u in uses if u["name"] == "search"]
        if self.search_tool and not searched and not exact:
            return self._tool("search", {"query": f"{subject['name']}", "limit": 5})
        hits = [(h.group(1), int(h.group(2)), int(h.group(3)), h.group(4)) for t in results for h in HIT.finditer(t)]
        for path, start, end, _ in hits:
            address = f"{path}:{start}-{end}"
            if address not in read and len(read) < self.max_reads:
                return self._tool("read", {"address": address})
        return self._give_up(template, subject, pieces or [])

    def _answer(self, template: str, subject: dict, found: tuple[object, str], ref: str, extra: tuple[str, str] | None = None) -> dict:
        answer, quote = found
        quotes = quote if isinstance(quote, list) else [quote]
        if self.v2:  # the subject and the value, plainly
            raised = [m.group(1).rsplit(".", 1)[-1] for q in quotes for m in [re.search(r"raise\s+([\w.]+)", q)] if m]
            text = {
                "default": f"`{subject['qual']}` parameter `{subject.get('param')}` defaults to `{answer}`",
                "constant": f"`{subject['qual']}` is `{answer}`",
                "raises": f"`{subject['qual']}` raises {', '.join(f'`{r}`' for r in dict.fromkeys(raised))} directly",
                "locate": f"`{subject['name']}` is defined in `{answer}`",
                "session": f"the {subject.get('attr')} of service `{subject['name']}` is `{answer}`",
            }[template]
        elif template in ("default", "constant"):
            text = f"The value is {answer}."
        elif template == "raises":
            text = "The function raises this exception directly."
        else:
            text = "The function is defined in the cited file."
        citations = [{"ref": ref, "quote": q} for q in quotes] + ([{"ref": extra[0], "quote": extra[1]}] if extra else [])
        return self._reply(self._format(answer, [{"text": text, "citations": citations}]))

    def _format(self, answer: object, claims: list[dict], abstain: bool = False) -> str:
        return (pilot_answer_text_v2 if self.v2 else pilot_answer_text)(answer, claims, abstain)

    def _summarize(self, request: dict) -> dict:
        text = "\n".join(t for m in request["messages"] for _, t in _text_blocks(m["content"]))
        refs = list(dict.fromkeys(re.findall(r"\[([^\]\n]+:\d+-\d+)\]", text)))
        names = list(dict.fromkeys(re.findall(r"def (\w+)", text) + re.findall(r"^([A-Z][A-Z0-9_]+) =", text, re.M)))
        return self._reply(f"Read {', '.join(refs) or 'nothing'}.\nNames seen: {', '.join(names[:40]) or 'none'}.")

    @staticmethod
    def _reply(text: str) -> dict:
        return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}

    def _tool(self, name: str, args: dict) -> dict:
        self.tool_calls += 1
        return {"content": [{"type": "tool_use", "id": f"toolu_fake_{self.tool_calls:05d}", "name": name, "input": args}], "stop_reason": "tool_use"}
